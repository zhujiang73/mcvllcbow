OpenCV LLC BOW 图像分类


OpenCV LLC BOW Win32 MinGW 版测试中...

关于 OpenCV LLC BOW 图像分类算法可以参考网上的一些资料：

BoW(SIFT/SURF/...)+SVM/KNN的OpenCV 实现:   http://www.tuicool.com/articles/VnARzi

Bag of Word闲谈:   http://blog.csdn.net/zjjconan/article/details/7843825

图像的稀疏表示——ScSPM和LLC的总结:   http://blog.csdn.net/jwh_bupt/article/details/9837555


原版的 LLC 是 Matlab code，我把编码部分改成了 OpenCV 的, 和原来的 OpenCV BOW 整合，但是目前还没做 Spatial Pyramid 分块处理。

开发环境是 MinGW Win32 CodeBlocks ，没有使用 Win 特有的 API ，移植应该比较容易。

编译完成后：

训练命令用法：
train: mcvllcbow.exe [train] [databaseDir] [resultDir]
example: mcvllcbow.exe train  ../data/train/  ../data/result/

测试命令用法：
test: mcvllcbow.exe [test] [sample_name] [test_dir] [svms_dir] [vocabulary_file]
example: mcvllcbow.exe test  sunflower  ../data/imgs/sunflower  ../data/result/svms  ../data/result/vocabulary.xml.gz

