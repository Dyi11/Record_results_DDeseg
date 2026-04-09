通过build_feabank_from_metadata_aobj.py生成DDeseg里面的avss_aud_bank: "/mnt/sdc/dy/data/Re_AVS/1s_k5_feabank.npy"

其中，KMeans 原型生成方式: "center" or "nearest", 设置MODE = "center"对应生成的是feabank1.npy， MODE = "nearest"对应生成的是feabank2.npy

feabank1.npy在HTSAT，v1m条件下获得/home/dy/DDESeg-main/output/test/20260408-214136/ck_epoch_46_maxmiou0.725_fscore0.815.pth

feabank2.npy在HTSAT，v1m条件下获得

通过check_mask.py发现这两种和原有的npy还是有差距的，在模型中的效果倒是区别不大。但是我自己生成的npy类和类之间的距离太近了，没有很好区分开，这个还需要探索如何改进。
