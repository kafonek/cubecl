//! Log-only instrumentation of the per-device task channel.
//!
//! Records how many tasks a device's task queue receives, how deep the ring buffer actually
//! gets, and how often a producer has to wait for a full buffer. Nothing here changes
//! scheduling, ordering or capacity: every function only touches counters and emits log
//! lines.
//!
//! Counters are a fixed array indexed by [`DeviceId::index_id`], so there is no allocation
//! and no lock on the submit path. Devices at an index at or above [`MAX_DEVICES`] are not
//! recorded.

use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::string::{String, ToString};

/// Highest device index the counters cover.
pub const MAX_DEVICES: usize = 16;

/// Number of enqueues between periodic summary lines.
const SUMMARY_EVERY: u64 = 8192;

macro_rules! counters_u64 {
    ($name:ident) => {
        static $name: [AtomicU64; MAX_DEVICES] = [const { AtomicU64::new(0) }; MAX_DEVICES];
    };
}

counters_u64!(ENQUEUED);
counters_u64!(FULL_WAITS);
counters_u64!(SWAPS);
counters_u64!(COLLECTIVES);
counters_u64!(ROUNDS);
counters_u64!(ENQUEUED_AT_LAST_ROUND);
counters_u64!(COLLECTIVES_AT_LAST_ROUND);
counters_u64!(FULL_WAITS_AT_LAST_ROUND);

/// Deepest slot index a real (non-padding) task has been written to, per device.
static PEAK_INDEX: [AtomicU32; MAX_DEVICES] = [const { AtomicU32::new(0) }; MAX_DEVICES];

fn slot(device_index: u16) -> Option<usize> {
    let index = device_index as usize;
    (index < MAX_DEVICES).then_some(index)
}

fn thread_name() -> String {
    std::thread::current()
        .name()
        .unwrap_or("unnamed")
        .to_string()
}

/// Records one task written into `device_index`'s ring buffer at slot `slot_index`.
///
/// Logs a line whenever the deepest slot ever used grows, which is monotone per device and
/// so bounded by the buffer size.
pub fn record_enqueue(device_index: u16, slot_index: usize, capacity: usize) {
    let Some(i) = slot(device_index) else {
        return;
    };

    let total = ENQUEUED[i].fetch_add(1, Ordering::Relaxed) + 1;
    let previous_peak = PEAK_INDEX[i].fetch_max(slot_index as u32, Ordering::Relaxed);

    if slot_index as u32 > previous_peak {
        log::info!(
            "chanocc peak device={device_index} slot_index={slot_index} capacity={capacity} \
             enqueued_total={total} thread={}",
            thread_name()
        );
    }

    if total.is_multiple_of(SUMMARY_EVERY) {
        log::info!(
            "chanocc summary device={device_index} enqueued_total={total} \
             peak_slot_index={} capacity={capacity} full_waits={} swaps={} collectives={}",
            PEAK_INDEX[i].load(Ordering::Relaxed),
            FULL_WAITS[i].load(Ordering::Relaxed),
            SWAPS[i].load(Ordering::Relaxed),
            COLLECTIVES[i].load(Ordering::Relaxed),
        );
    }
}

/// Records one producer stall: an `enqueue` that found the buffer full and had to back off.
///
/// Called once per stalled `enqueue` call, not once per retry.
pub fn record_full_wait(device_index: u16) {
    if let Some(i) = slot(device_index) {
        let waits = FULL_WAITS[i].fetch_add(1, Ordering::Relaxed) + 1;
        if waits == 1 || waits.is_multiple_of(64) {
            log::info!(
                "chanocc full_wait device={device_index} full_waits={waits} thread={}",
                thread_name()
            );
        }
    }
}

/// Records one buffer swap by the device's server thread.
pub fn record_swap(device_index: u16) {
    if let Some(i) = slot(device_index) {
        SWAPS[i].fetch_add(1, Ordering::Relaxed);
    }
}

/// Records one collective submitted to `device_index`.
pub fn record_collective(device_index: u16) {
    if let Some(i) = slot(device_index) {
        COLLECTIVES[i].fetch_add(1, Ordering::Relaxed);
    }
}

/// Marks the end of one synchronized round on `device_index` and logs what the round cost.
///
/// The deltas are what size a buffer against: `enqueued` is every task the round put in the
/// queue for this device, `collectives` is how many of them were collectives.
pub fn mark_round(device_index: u16) {
    let Some(i) = slot(device_index) else {
        return;
    };

    let round = ROUNDS[i].fetch_add(1, Ordering::Relaxed);

    let enqueued = ENQUEUED[i].load(Ordering::Relaxed);
    let collectives = COLLECTIVES[i].load(Ordering::Relaxed);
    let full_waits = FULL_WAITS[i].load(Ordering::Relaxed);

    let enqueued_delta = enqueued - ENQUEUED_AT_LAST_ROUND[i].swap(enqueued, Ordering::Relaxed);
    let collectives_delta =
        collectives - COLLECTIVES_AT_LAST_ROUND[i].swap(collectives, Ordering::Relaxed);
    let full_waits_delta =
        full_waits - FULL_WAITS_AT_LAST_ROUND[i].swap(full_waits, Ordering::Relaxed);

    log::info!(
        "chanocc round device={device_index} round={round} enqueued={enqueued_delta} \
         collectives={collectives_delta} full_waits={full_waits_delta} \
         peak_slot_index={} swaps={}",
        PEAK_INDEX[i].load(Ordering::Relaxed),
        SWAPS[i].load(Ordering::Relaxed),
    );
}
