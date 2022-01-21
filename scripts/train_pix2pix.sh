set -ex
# python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --netG unet_256 --direction BtoA --lambda_L1 100 --dataset_mode aligned --norm batch --pool_size 0
python train.py --dataroot ./datasets/edges2shoes --name edges2shoes_pix2pix --model pix2pix --netG resnet_6blocks --direction AtoA --lambda_L1 100 --dataset_mode aligned --norm batch --pool_size 0
