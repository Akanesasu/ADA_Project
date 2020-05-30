运行我DCGAN、VAE和RealNVP的训练部分的方法如下：

```shell
python main.py
```

没有写test部分，如果有要求的话可以写一个，模型文件存储在本地电脑上。



运行参考的realNVP_Ref (来自https://github.com/fmu2/realNVP) 的训练部分的方法如下：

```
python train.py --dataset=celeba --batch_size=64 --base_dim=32 --res_blocks=2 --max_epoch=5
```

模型文件链接：

https://disk.pku.edu.cn:443/link/6D85133A510D8FFAF27E5D4046CA1A7B
有效期限：2020-07-28 23:59