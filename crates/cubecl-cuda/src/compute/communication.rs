use std::{collections::HashMap, sync::OnceLock};

use cubecl_core::{
    device::DeviceId,
    ir::ElemType,
    server::{CommunicationId, ReduceOperation},
};
use cubecl_environment::sync::Mutex;

/// Global state map from [`CommunicationId`] to boxed [`cudarc::nccl::sys::ncclUniqueId`].
static UNIQUE_IDS_MAP: OnceLock<Mutex<HashMap<CommunicationId, cudarc::nccl::sys::ncclUniqueId>>> =
    OnceLock::new();

/// Hex-encoded [`cudarc::nccl::sys::ncclUniqueId`] shared by every process in a multi-process job.
pub const UNIQUE_ID_VAR: &str = "CUBECL_NCCL_UNIQUE_ID";
/// Index of this process among [`WORLD_PROCESSES_VAR`] processes.
pub const PROCESS_RANK_VAR: &str = "CUBECL_PROCESS_RANK";
/// Number of processes taking part in the collective.
pub const WORLD_PROCESSES_VAR: &str = "CUBECL_WORLD_PROCESSES";

/// Placement of this process inside a collective that spans several processes.
///
/// `local_devices` is the number of devices this process owns. Every process must own the same
/// number, because the rank of a device is its local position offset by the process rank.
///
/// Both values fall back to a single process that owns every device in the group.
pub(crate) fn global_placement(local_devices: usize) -> (usize, usize) {
    let process_rank = env_usize(PROCESS_RANK_VAR).unwrap_or(0);
    let world_processes = env_usize(WORLD_PROCESSES_VAR).unwrap_or(1);
    (
        process_rank * local_devices,
        world_processes * local_devices,
    )
}

fn env_usize(key: &str) -> Option<usize> {
    std::env::var(key).ok()?.parse().ok()
}

pub(crate) fn get_nccl_comm_id(device_ids: Vec<DeviceId>) -> cudarc::nccl::sys::ncclUniqueId {
    if let Some(id) = unique_id_from_env() {
        return id;
    }

    let mut unique_ids_map = UNIQUE_IDS_MAP.get_or_init(Default::default).lock();
    let comm_id = CommunicationId::from(device_ids);
    match unique_ids_map.get_mut(&comm_id) {
        Some(id) => *id,
        None => {
            let id = cudarc::nccl::result::get_uniqueid().unwrap();
            unique_ids_map.insert(comm_id, id);
            id
        }
    }
}

/// NCCL embeds rank 0's bootstrap address in the unique ID, so a process cannot generate an ID that
/// another process can join. Rank 0 must publish the ID it created, and every other process must
/// receive it through [`UNIQUE_ID_VAR`].
fn unique_id_from_env() -> Option<cudarc::nccl::sys::ncclUniqueId> {
    let hex = std::env::var(UNIQUE_ID_VAR).ok()?;
    let bytes = decode_hex(&hex)
        .unwrap_or_else(|| panic!("{UNIQUE_ID_VAR} must be hex, got {} chars", hex.len()));
    assert_eq!(
        bytes.len(),
        128,
        "{UNIQUE_ID_VAR} must decode to 128 bytes, got {}",
        bytes.len()
    );

    let mut id = cudarc::nccl::sys::ncclUniqueId { internal: [0; 128] };
    for (slot, byte) in id.internal.iter_mut().zip(bytes) {
        *slot = byte as _;
    }
    Some(id)
}

fn decode_hex(hex: &str) -> Option<Vec<u8>> {
    if !hex.len().is_multiple_of(2) {
        return None;
    }
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).ok())
        .collect()
}

pub(crate) fn to_nccl_op(op: ReduceOperation) -> cudarc::nccl::sys::ncclRedOp_t {
    match op {
        ReduceOperation::Sum => cudarc::nccl::sys::ncclRedOp_t::ncclSum,
        ReduceOperation::Mean => cudarc::nccl::sys::ncclRedOp_t::ncclAvg,
    }
}

pub(crate) fn get_nccl_dtype_count(
    dtype: ElemType,
    size: u64,
) -> (cudarc::nccl::sys::ncclDataType_t, usize) {
    match dtype {
        ElemType::Index => panic!("Index not supported in NCCL"),
        ElemType::Float(
            cubecl_core::ir::FloatKind::E2M1
            | cubecl_core::ir::FloatKind::E2M1x2
            | cubecl_core::ir::FloatKind::E2M3
            | cubecl_core::ir::FloatKind::E3M2
            | cubecl_core::ir::FloatKind::UE8M0,
        ) => panic!("Minifloat not supported in NCCL"),
        ElemType::Float(cubecl_core::ir::FloatKind::E4M3) => (
            cudarc::nccl::sys::ncclDataType_t::ncclFloat8e4m3,
            size as usize,
        ),
        ElemType::Float(cubecl_core::ir::FloatKind::E5M2) => (
            cudarc::nccl::sys::ncclDataType_t::ncclFloat8e5m2,
            size as usize,
        ),
        ElemType::Float(cubecl_core::ir::FloatKind::F16) => (
            cudarc::nccl::sys::ncclDataType_t::ncclFloat16,
            (size / 2) as usize,
        ),
        ElemType::Float(cubecl_core::ir::FloatKind::BF16) => (
            cudarc::nccl::sys::ncclDataType_t::ncclBfloat16,
            (size / 2) as usize,
        ),
        ElemType::Float(cubecl_core::ir::FloatKind::Flex32) => {
            panic!("NCCL doesn't support Flex32 format.")
        }

        ElemType::Float(cubecl_core::ir::FloatKind::F32) => (
            cudarc::nccl::sys::ncclDataType_t::ncclFloat32,
            (size / 4) as usize,
        ),
        ElemType::Float(cubecl_core::ir::FloatKind::TF32) => {
            panic!("NCCL doesn't support TF32 format.")
        }
        ElemType::Float(cubecl_core::ir::FloatKind::F64) => (
            cudarc::nccl::sys::ncclDataType_t::ncclFloat64,
            (size / 8) as usize,
        ),
        ElemType::Int(int_kind) => match int_kind {
            cubecl_core::ir::IntKind::I8 => {
                (cudarc::nccl::sys::ncclDataType_t::ncclInt8, size as usize)
            }
            cubecl_core::ir::IntKind::I16 => panic!("NCCL doesn't support Int16 format."),
            cubecl_core::ir::IntKind::I32 => (
                cudarc::nccl::sys::ncclDataType_t::ncclInt32,
                (size / 4) as usize,
            ),
            cubecl_core::ir::IntKind::I64 => (
                cudarc::nccl::sys::ncclDataType_t::ncclInt64,
                (size / 8) as usize,
            ),
        },
        ElemType::UInt(uint_kind) => match uint_kind {
            cubecl_core::ir::UIntKind::U8 => {
                (cudarc::nccl::sys::ncclDataType_t::ncclUint8, size as usize)
            }
            cubecl_core::ir::UIntKind::U16 => panic!("NCCL doesn't support UInt16 format."),
            cubecl_core::ir::UIntKind::U32 => (
                cudarc::nccl::sys::ncclDataType_t::ncclUint32,
                (size / 4) as usize,
            ),
            cubecl_core::ir::UIntKind::U64 => (
                cudarc::nccl::sys::ncclDataType_t::ncclUint64,
                (size / 8) as usize,
            ),
        },
        ElemType::Bool => panic!("NCCL doesn't support Bool format."),
    }
}
