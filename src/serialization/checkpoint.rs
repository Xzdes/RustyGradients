///! Checkpoint management utilities
///!
///! Manages model checkpoints with automatic cleanup of old checkpoints.

use super::{ModelMetadata, CheckpointInfo};
use crate::error::{Result, RustyGradientsError};
use crate::tensor::Tensor;
use std::fs;
use std::path::{Path, PathBuf};

#[cfg(feature = "serialization")]
use super::safetensors_format;

/// Checkpoint manager - автоматически управляет сохранением и очисткой checkpoints
pub struct CheckpointManager {
    pub checkpoint_dir: PathBuf,
    pub max_checkpoints: usize,
    pub keep_best: bool,
}

impl CheckpointManager {
    /// Create new checkpoint manager
    pub fn new(checkpoint_dir: impl Into<PathBuf>, max_checkpoints: usize) -> Self {
        let dir = checkpoint_dir.into();

        // Create directory if not exists
        if !dir.exists() {
            fs::create_dir_all(&dir).ok();
        }

        Self {
            checkpoint_dir: dir,
            max_checkpoints,
            keep_best: true,
        }
    }

    /// Save checkpoint with automatic cleanup
    #[cfg(feature = "serialization")]
    pub fn save_checkpoint(
        &self,
        weights: &[Tensor],
        weight_names: &[String],
        metadata: &ModelMetadata,
        step: usize,
        loss: f32,
    ) -> Result<PathBuf> {
        // Create checkpoint filename
        let checkpoint_path = self.checkpoint_dir.join(format!("checkpoint_step_{:06}.safetensors", step));

        // Save using safetensors
        safetensors_format::save_model(&checkpoint_path, weights, weight_names, metadata)?;

        // Save checkpoint info
        let info = CheckpointInfo {
            step,
            loss,
            timestamp: chrono::Utc::now().to_rfc3339(),
        };
        let info_path = checkpoint_path.with_extension("safetensors.info.json");
        let info_json = serde_json::to_string_pretty(&info)
            .map_err(|e| RustyGradientsError::SerializationError(e.to_string()))?;
        fs::write(info_path, info_json)
            .map_err(|e| RustyGradientsError::IoError(e.to_string()))?;

        // Cleanup old checkpoints
        self.cleanup_old_checkpoints()?;

        Ok(checkpoint_path)
    }

    /// Save checkpoint using legacy JSON format
    pub fn save_checkpoint_json(
        &self,
        weights: &[Tensor],
        metadata: &ModelMetadata,
        step: usize,
        loss: f32,
    ) -> Result<PathBuf> {
        let checkpoint_path = self.checkpoint_dir.join(format!("checkpoint_step_{:06}.json", step));

        super::json::save_json(&checkpoint_path, weights, metadata, step, loss)?;

        self.cleanup_old_checkpoints()?;

        Ok(checkpoint_path)
    }

    /// Cleanup old checkpoints, keeping only last N and best
    fn cleanup_old_checkpoints(&self) -> Result<()> {
        // Get all checkpoint files
        let mut checkpoints = self.list_checkpoints()?;

        if checkpoints.len() <= self.max_checkpoints {
            return Ok(());
        }

        // Sort by step (descending)
        checkpoints.sort_by(|a, b| b.step.cmp(&a.step));

        // Find best checkpoint (lowest loss)
        let best_idx = if self.keep_best {
            checkpoints
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.loss.partial_cmp(&b.loss).unwrap())
                .map(|(idx, _)| idx)
        } else {
            None
        };

        // Delete old checkpoints
        for (idx, checkpoint) in checkpoints.iter().enumerate().skip(self.max_checkpoints) {
            // Don't delete best checkpoint
            if let Some(best) = best_idx {
                if idx == best {
                    continue;
                }
            }

            // Delete checkpoint files
            if checkpoint.path.exists() {
                fs::remove_file(&checkpoint.path)
                    .map_err(|e| RustyGradientsError::IoError(e.to_string()))?;
            }

            // Delete metadata
            let metadata_path = checkpoint.path.with_extension("safetensors.json");
            if metadata_path.exists() {
                fs::remove_file(metadata_path).ok();
            }

            // Delete info
            let info_path = checkpoint.path.with_extension("safetensors.info.json");
            if info_path.exists() {
                fs::remove_file(info_path).ok();
            }
        }

        Ok(())
    }

    /// List all checkpoints in directory
    pub fn list_checkpoints(&self) -> Result<Vec<CheckpointEntry>> {
        let mut checkpoints = Vec::new();

        let entries = fs::read_dir(&self.checkpoint_dir)
            .map_err(|e| RustyGradientsError::IoError(e.to_string()))?;

        for entry in entries {
            let entry = entry.map_err(|e| RustyGradientsError::IoError(e.to_string()))?;
            let path = entry.path();

            // Only .safetensors files (not .json metadata)
            if path.extension().and_then(|s| s.to_str()) == Some("safetensors") {
                // Parse step from filename
                if let Some(step) = parse_step_from_filename(&path) {
                    // Try to load checkpoint info
                    let info_path = path.with_extension("safetensors.info.json");
                    let (loss, timestamp) = if info_path.exists() {
                        let info_json = fs::read_to_string(info_path)
                            .map_err(|e| RustyGradientsError::IoError(e.to_string()))?;
                        let info: CheckpointInfo = serde_json::from_str(&info_json)
                            .map_err(|e| RustyGradientsError::SerializationError(e.to_string()))?;
                        (info.loss, info.timestamp)
                    } else {
                        (f32::INFINITY, String::new())
                    };

                    checkpoints.push(CheckpointEntry {
                        path,
                        step,
                        loss,
                        timestamp,
                    });
                }
            }
        }

        Ok(checkpoints)
    }

    /// Load latest checkpoint
    #[cfg(feature = "serialization")]
    pub fn load_latest(&self) -> Result<(Vec<Vec<f32>>, Vec<Vec<usize>>, Vec<String>, ModelMetadata)> {
        let checkpoints = self.list_checkpoints()?;

        if checkpoints.is_empty() {
            return Err(RustyGradientsError::IoError("No checkpoints found".to_string()));
        }

        // Get latest checkpoint
        let latest = checkpoints
            .iter()
            .max_by_key(|c| c.step)
            .ok_or_else(|| RustyGradientsError::IoError("No checkpoints found".to_string()))?;

        safetensors_format::load_model(&latest.path)
    }

    /// Load best checkpoint (lowest loss)
    #[cfg(feature = "serialization")]
    pub fn load_best(&self) -> Result<(Vec<Vec<f32>>, Vec<Vec<usize>>, Vec<String>, ModelMetadata)> {
        let checkpoints = self.list_checkpoints()?;

        if checkpoints.is_empty() {
            return Err(RustyGradientsError::IoError("No checkpoints found".to_string()));
        }

        let best = checkpoints
            .iter()
            .min_by(|a, b| a.loss.partial_cmp(&b.loss).unwrap())
            .ok_or_else(|| RustyGradientsError::IoError("No checkpoints found".to_string()))?;

        safetensors_format::load_model(&best.path)
    }
}

/// Checkpoint entry metadata
#[derive(Debug, Clone)]
pub struct CheckpointEntry {
    pub path: PathBuf,
    pub step: usize,
    pub loss: f32,
    pub timestamp: String,
}

/// Parse step number from checkpoint filename
fn parse_step_from_filename(path: &Path) -> Option<usize> {
    path.file_stem()
        .and_then(|s| s.to_str())
        .and_then(|s| {
            // Format: checkpoint_step_000123
            if let Some(step_str) = s.strip_prefix("checkpoint_step_") {
                step_str.parse::<usize>().ok()
            } else {
                None
            }
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_step() {
        let path = PathBuf::from("checkpoint_step_000123.safetensors");
        assert_eq!(parse_step_from_filename(&path), Some(123));

        let path = PathBuf::from("checkpoint_step_001000.safetensors");
        assert_eq!(parse_step_from_filename(&path), Some(1000));
    }
}
