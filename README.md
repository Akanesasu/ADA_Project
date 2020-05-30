运行我DCGAN、VAE和RealNVP的训练部分的方法如下：

```shell
python main.py
```

没有写test部分，如果有要求的话可以写一个，模型文件存储在本地电脑上。



运行参考的realNVP_Ref的训练部分的方法如下：

```
python train.py --dataset=celeba --batch_size=64 --base_dim=32 --res_blocks=2 --max_epoch=5
```

