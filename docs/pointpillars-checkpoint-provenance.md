# PointPillars checkpoint provenance and rights review (RV-020)

Audit date: **2026-11-14**. Technical provenance is verified; an explicit license
statement identifying this externally hosted checkpoint was **not found in the
reviewed sources**. This audit does not declare the weights Apache-2.0, CC-licensed,
public domain, or cleared for commercial use/redistribution. It also does not
conclude that a license cannot exist elsewhere.

## Verified source, bytes and configuration

The official [OpenPCDet KITTI Model Zoo](https://github.com/open-mmlab/OpenPCDet/blob/233f849829b6ac19afb8af8837a0246890908755/README.md#kitti-3d-object-detection-baselines)
links its **PointPillar** row to both the KITTI model YAML and Google Drive file
ID `1wMxWTpU1qUoY3DsCH31WJmvJxcjFXKlm`. These links identify the intended pair;
they do not establish an additional weight-license grant.

| Item | Verified value |
| --- | --- |
| Official repository | `open-mmlab/OpenPCDet` |
| Pinned full revision | `233f849829b6ac19afb8af8837a0246890908755` |
| Upstream default branch at review | `master`, resolving to that same revision |
| Official checkpoint link | [Google Drive file](https://drive.google.com/file/d/1wMxWTpU1qUoY3DsCH31WJmvJxcjFXKlm/view) |
| Provider filename | `pointpillar_7728.pth` |
| Raven experiment's local alias | `pointpillar.pth` (filename alone does not identify bytes) |
| File size | **19,383,576 bytes** |
| Checkpoint SHA256 | `4c83fc0fa02575b9b3e9dec676f698e7a70bb5a795e89f91df8a96b916fa19e2` |
| Model configuration | [`tools/cfgs/kitti_models/pointpillar.yaml`](https://github.com/open-mmlab/OpenPCDet/blob/233f849829b6ac19afb8af8837a0246890908755/tools/cfgs/kitti_models/pointpillar.yaml) |
| Model YAML SHA256 | `170a9ffe76cfd8509d1044cfbcf1cbd44c5d320fda81bf0089a8d5efaf1c91c8` |
| Referenced base YAML | [`tools/cfgs/dataset_configs/kitti_dataset.yaml`](https://github.com/open-mmlab/OpenPCDet/blob/233f849829b6ac19afb8af8837a0246890908755/tools/cfgs/dataset_configs/kitti_dataset.yaml) |
| Base YAML SHA256 | `a6c44a7b0f15b94e135946cecbd8b5541906a69d37811705e9884a411e7b1797` |

A fresh unauthenticated download from the public Drive download endpoint on the
audit date returned HTTP 200, `application/octet-stream`, and a Content-Disposition
filename of `pointpillar_7728.pth`. Hashing the new temporary copy matched the
recorded experiment checksum exactly. The earlier temporary checkpoint was also
rehashed read-only and matched. Neither file was loaded into torch, deserialized,
modified or added to Raven. The public Drive view named the file but exposed no
checkpoint license in the reviewed response. This does not inspect account-only
or otherwise unavailable Drive metadata.

The YAML defines `PointPillar`, class order Car/Pedestrian/Cyclist and the KITTI
XYZI base configuration. Its linked source revision/config pair is verified, not
a reconstruction of an unarchived historical notebook or proof of every training
sample/override. Retain the resolved runtime config for each experiment. Existing
T4 execution and numerical-comparison evidence is in the
[feasibility](pointpillars-feasibility.md) and [reference](reference-validation.md)
guides; successful inference does not establish license rights or KITTI accuracy.

## Three distinct licensing questions

| Material | Primary evidence | Finding and boundary |
| --- | --- | --- |
| OpenPCDet project/source | [Root LICENSE](https://github.com/open-mmlab/OpenPCDet/blob/233f849829b6ac19afb8af8837a0246890908755/LICENSE), [README license section](https://github.com/open-mmlab/OpenPCDet/blob/233f849829b6ac19afb8af8837a0246890908755/README.md#license), [`setup.py`](https://github.com/open-mmlab/OpenPCDet/blob/233f849829b6ac19afb8af8837a0246890908755/setup.py) | Apache-2.0 is explicitly declared. Its grants/conditions apply to the covered work; preserve applicable notices and license obligations when reusing/distributing covered code. Dependencies and separately licensed components need their own review. |
| This pretrained checkpoint | Official Model Zoo download association and the byte identity above | No checkpoint-specific license/notice or explicit statement extending the project's Apache grant to this Drive file was located. Whether the general project declaration covers these externally hosted bytes remains unresolved. |
| KITTI data/benchmark | [Official copyright/citation notice](https://www.cvlibs.net/datasets/kitti/), [CC BY-NC-SA 3.0 deed](https://creativecommons.org/licenses/by-nc-sa/3.0/), [legal code](https://creativecommons.org/licenses/by-nc-sa/3.0/legalcode) | KITTI states Attribution-NonCommercial-ShareAlike 3.0. Attribution, noncommercial use and applicable share-alike conditions concern that licensed material. The checkpoint's legal status is not established merely by naming its training dataset. |

The pinned tree also contains an [MIT license for the KITTI Python evaluator](https://github.com/open-mmlab/OpenPCDet/blob/233f849829b6ac19afb8af8837a0246890908755/pcdet/datasets/kitti/kitti_object_eval_python/LICENSE).
That notice concerns evaluator software, not a license declaration for KITTI raw
data or PointPillar weights. Similarly, the project's broad Apache declaration
is relevant evidence, but this audit cannot confidently resolve its scope over
the external binary. Do not describe the binary as definitely excluded from Apache
or definitely covered by it without further support.

KITTI [download/submission policy](https://www.cvlibs.net/datasets/kitti/user_login.php)
requires registration and declared usage purpose; authenticated account-specific
terms were not reviewed. CC BY-NC-SA provides conditional sharing permissions,
not unrestricted redistribution. Raven's policy is to distribute no dataset or
checkpoint payloads. This is a project policy, not a claim that every redistribution
is legally prohibited. This audit does not decide whether particular trained
weights, fine-tuned weights or outputs are derivatives of KITTI data, or which
third-party rights apply to them. Neither license is automatically assigned to
the other asset. See the [dataset policy](kitti-dataset.md).

## Official statements checked and limits of the search

- The pinned README Model Zoo, root license, setup metadata, model/base YAMLs,
  [`docs/DEMO.md`](https://github.com/open-mmlab/OpenPCDet/blob/233f849829b6ac19afb8af8837a0246890908755/docs/DEMO.md)
  and [`docs/GETTING_STARTED.md`](https://github.com/open-mmlab/OpenPCDet/blob/233f849829b6ac19afb8af8837a0246890908755/docs/GETTING_STARTED.md)
  were inspected. Demo/testing instructions invite downloading and running
  pretrained models, including on custom scans. This documents intended technical
  workflows; it does not expressly resolve commercial, redistribution or
  derivative-weight permissions for this particular binary.
- The complete, non-truncated recursive Git tree at the pinned revision contained
  two license/notice/model-card/terms filename matches: root `LICENSE` and the
  evaluator's nested `LICENSE`. No separately named checkpoint license, NOTICE
  or model card was found by that search. Current master is the same revision;
  there were no published GitHub releases/assets to inspect at review time.
- Official-repository issue searches for `license`, `license in:title`,
  `licensing`, `commercial`, `weights license` and `pretrained commercial`, plus
  targeted web searches for pretrained-weight terms, yielded no reviewed
  authoritative KITTI PointPillar-specific permission statement. Many broad
  license hits were installation logs. This is a bounded search, not proof that
  every historic comment, private agreement or unindexed page lacks such terms.
- The [README's Waymo section](https://github.com/open-mmlab/OpenPCDet/blob/233f849829b6ac19afb8af8837a0246890908755/README.md#waymo-open-dataset-baselines)
  and an [upstream collaborator response](https://github.com/open-mmlab/OpenPCDet/issues/1205#issuecomment-1334685307)
  explicitly discuss Waymo licensing constraints on releasing those models.
  These statements concern Waymo models, not this KITTI checkpoint. They neither
  grant KITTI weight permissions nor transfer Waymo restrictions to KITTI.
- The downloaded binary is legacy pickle-based serialization, not a ZIP with a
  browsable license sidecar. Embedded serialized metadata was not deserialized;
  absence of an embedded notice is **not** claimed. No unsafe loading was used
  merely to inspect potential licensing metadata.

No upstream clarification request was sent and no private permission was inferred.

## Supported workflow versus permission clearance

| Proposed activity | What this review supports |
| --- | --- |
| Reproduce the file/config/revision identity | Supported by official association, independent byte hashes and reproducible checks below. |
| Run the provider's demo/test workflow locally for research | Technically documented upstream and demonstrated in Raven's earlier experiments. An explicit checkpoint-license determination is still missing; research intent alone is not a substitute for applicable permission/terms. |
| Use covered OpenPCDet code | Governed by its Apache-2.0 terms and other applicable component rights, independently of dataset/weight permissions. |
| Commercial checkpoint deployment | No commercial clearance is established by this audit. Obtain an authoritative scope/permission determination before treating it as cleared. |
| Bundle/mirror/relicense the weights, including in releases or containers | Permission has not been established here. Raven does not distribute them; users obtain their own copy through the provider. |
| Redistribute fine-tuned weights or dataset-derived assets | Not resolved. Confirm rights/conditions for each relevant asset; this review supplies no new grant. |

Questions for an authorized upstream clarification: does Apache-2.0 cover the
exact Drive ID/checksum, or is another license intended? What conditions govern
noncommercial research inference, commercial deployment, mirroring/bundling,
fine-tuning/redistributing modified weights, and attribution? Are there relevant
KITTI or other third-party conditions? Archive a dated, attributable response or
checkpoint-specific license referencing the exact asset before changing this
status. The questions are documented, not posted on the user's behalf.

## Repository exclusion and repeatable checks

At Raven main `bcd40129f5eee4a95bec2a7a988258f07a1fc906`, no tracked
`.pth/.pt/.ckpt/.onnx/.safetensors/.h5` files or tracked blobs over 1 MB were found.
The known 19,383,576-byte checkpoint is therefore not a tracked payload under a
different filename either. A model-extension history search across reachable
local refs returned no commits. Raven's GitHub releases list was empty, so no
release asset distributes it. This confirms the checked source/history/releases,
not every possible external mirror or unpublished local package.

`.gitignore` excludes `/weights/`, `/checkpoints/` and `*.pth` (among other model
formats). Ignore rules alone are not the evidence of non-tracking; the Git checks
above establish the current state. All audit downloads stayed in temporary storage
outside the repository. No weights or license changes are added by this PR.

Read-only checks, using your actual trusted local paths:

```bash
shasum -a 256 /path/to/pointpillar.pth
# Expected: 4c83fc0fa02575b9b3e9dec676f698e7a70bb5a795e89f91df8a96b916fa19e2

git -C /path/to/OpenPCDet rev-parse HEAD
shasum -a 256 /path/to/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml
shasum -a 256 /path/to/OpenPCDet/tools/cfgs/dataset_configs/kitti_dataset.yaml

git ls-files '*.pth' '*.pt' '*.ckpt' '*.onnx' '*.safetensors' '*.h5'
git check-ignore -v weights/pointpillar.pth checkpoints/pointpillar.pth pointpillar_7728.pth
```

Do not deserialize a checkpoint to calculate its checksum. Raven's existing safe
loading defaults and explicit trust flag are unchanged by this documentation review.

## Acceptance review and recommendation

| RV-020 requirement | Status |
| --- | --- |
| Checksum and matching model config verified | Satisfied; exact hashes, official pairing and full revision are recorded. |
| Weights remain untracked | Satisfied for checked source/history/releases; payloads stay outside Raven. |
| Relevant commands/research evidence documented | Addressed by this documentation PR; review/merge remains the owner's decision. |
| Record applicable terms | Code/dataset terms documented; checkpoint scope was investigated but remains unresolved. |

**Recommendation: keep RV-020 open pending authoritative checkpoint-rights
clarification.** The provenance audit itself is complete, and the literal
technical acceptance criteria have evidence. Closing the issue as if weight
permissions were settled would overstate the result. The owner could instead
explicitly re-scope it to a completed provenance audit and separately track the
unresolved rights question; that is a scope decision, not a finding of permission.
This PR neither closes RV-020 nor creates an upstream request, and should not be
merged automatically. No model, checkpoint, inference path or project license is
changed.
