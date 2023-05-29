from rknn.api import RKNN



PT_PATH = r'/workspaces/rknn-toolkit/nanodet-service/nanodet-plus-m_416_torchscript.pt'
# 创建 RKNN 对象
rknn = RKNN(verbose=True)
# 设置模型输入预处理参数
rknn.config()# mean_values=[[123.675, 116.28, 103.53]], std_values=[[58.395, 58.395,58.395]], reorder_channel='0 1 2'
# 加载 PyTorch 模型
ret = rknn.load_pytorch(model=PT_PATH, input_size_list=[[3,416,416]])
if ret != 0:
        print('Load Pytorch model failed!')
        exit(ret)
# 构建 RKNN 量化模型
ret = rknn.build(do_quantization=False,dataset='/workspaces/rknn-toolkit/nanodet-service/dataset.txt')#do_quantization=True, dataset='./dataset.txt'
if ret != 0:
        print('Build model failed!')
        exit(ret)
# 导出 RKNN 模型到指定路径
ret = rknn.export_rknn(r'/workspaces/rknn-toolkit/nanodet-service/nanodet-plus-m_416_torchscript.rknn')
if ret != 0:
        print('Export nanodet-plus-m_416_torchscript.rknn failed!')
        exit(ret)
rknn.release()