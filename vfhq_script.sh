# merge duc annotation
prata specifics research facegen vfhq-posemerge-multithread /lustre/scratch/client/scratch/research/group/anhgroup/ducvh5/VFHQ_v2/test/synergy /lustre/scratch/client/scratch/research/group/anhgroup/trungdt21/data/vfhq_v2/pose_txt_anh/test /lustre/scratch/client/scratch/research/group/anhgroup/ducvh5/VFHQ_v2/test/iqa_txt /lustre/scratch/client/scratch/research/group/anhgroup/ducvh5/VFHQ_v2/test/txt vfhqv2_testmerge_gt_synergy_poseanh --workers 16
prata specifics research facegen vfhq-posemerge-multithread /lustre/scratch/client/scratch/research/group/anhgroup/ducvh5/VFHQ_v2/train/synergy /lustre/scratch/client/scratch/research/group/anhgroup/trungdt21/data/vfhq_v2/pose_txt_anh/train /lustre/scratch/client/scratch/research/group/anhgroup/ducvh5/VFHQ_v2/train/iqa_txt /lustre/scratch/client/scratch/research/group/anhgroup/ducvh5/VFHQ_v2/train/txt vfhqv2_trainmerge_gt_synergy_poseanh --workers 16

# combine multiid to one id per csv
prata specifics research facegen vfhq-combine-multiid-into-one vfhqv2_testmerge_gt_synergy_poseanh /lustre/scratch/client/scratch/research/group/anhgroup/trungdt21/data/vfhq_v2/version2/merge_synergy_anhpose_iqa_anno/uniqueid/test
prata specifics research facegen vfhq-combine-multiid-into-one vfhqv2_trainmerge_gt_synergy_poseanh /lustre/scratch/client/scratch/research/group/anhgroup/trungdt21/data/vfhq_v2/version2/merge_synergy_anhpose_iqa_anno/uniqueid/train

# merge with directmhp
prata specifics research facegen vfhq-directmhp-merge /lustre/scratch/client/scratch/research/group/anhgroup/trungdt21/data/pose_preds/DirectMHP/vfhq_infered/final/test /lustre/scratch/client/scratch/research/group/anhgroup/trungdt21/data/vfhq_v2/version2/merge_synergy_anhpose_iqa_anno/uniqueid/test /lustre/scratch/client/scratch/research/group/anhgroup/trungdt21/data/vfhq_v2/version2/merge_all/test --iou-thresh 0.1
prata specifics research facegen vfhq-directmhp-merge /lustre/scratch/client/scratch/research/group/anhgroup/trungdt21/data/pose_preds/DirectMHP/vfhq_infered/final/train /lustre/scratch/client/scratch/research/group/anhgroup/trungdt21/data/vfhq_v2/version2/merge_synergy_anhpose_iqa_anno/uniqueid/train /lustre/scratch/client/scratch/research/group/anhgroup/trungdt21/data/vfhq_v2/version2/merge_all/train --iou-thresh 0.1
