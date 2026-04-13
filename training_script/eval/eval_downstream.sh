cd src

model=$1
pretrained=$2

current=`date "+%Y-%m-%d-%H:%M:%S-"`
dir=eval_results/$current$model
mkdir eval_results/$current$model

# zero-shot retrieval
clip_benchmark eval --dataset=mscoco_captions --task=zeroshot_retrieval --pretrained=$pretrained --model=$model --output=$dir/result_coco.json --batch_size=64 --dataset_root ./dataset/downstream/coco/
clip_benchmark eval --dataset=flickr30k --task=zeroshot_retrieval --pretrained=$pretrained --model=$model --output=$dir/result_flickr30k.json --batch_size=64 --dataset_root ./dataset/downstream/flickr30k/

# # zero-shot classification
# clip_benchmark eval --dataset=cifar10 --task=zeroshot_classification --pretrained=$pretrained --model=$model --output=$dir/result_cifar10.json --batch_size=64 --dataset_root ./dataset/downstream/cifar10/
# clip_benchmark eval --dataset=cifar100 --task=zeroshot_classification --pretrained=$pretrained --model=$model --output=$dir/result_cifar100.json --batch_size=64 --dataset_root ./dataset/downstream/cifar100/
# clip_benchmark eval --dataset=food101 --task=zeroshot_classification --pretrained=$pretrained --model=$model --output=$dir/result_food101.json --batch_size=64 --dataset_root ./dataset/downstream/food101/
# clip_benchmark eval --dataset=sun397 --task=zeroshot_classification --pretrained=$pretrained --model=$model --output=$dir/result_sun397.json --batch_size=64 --dataset_root ./dataset/downstream/sun397/
# clip_benchmark eval --dataset=cars --task=zeroshot_classification --pretrained=$pretrained --model=$model --output=$dir/result_cars.json --batch_size=64 --dataset_root ./dataset/downstream/cars/
# clip_benchmark eval --dataset=fgvc_aircraft --task=zeroshot_classification --pretrained=$pretrained --model=$model --output=$dir/result_aircraft.json --batch_size=64 --dataset_root ./dataset/downstream/aircraft/
# clip_benchmark eval --dataset=dtd --task=zeroshot_classification --pretrained=$pretrained --model=$model --output=$dir/result_dtd.json --batch_size=64 --dataset_root ./dataset/downstream/dtd/
# clip_benchmark eval --dataset=pets --task=zeroshot_classification --pretrained=$pretrained --model=$model --output=$dir/result_pets.json --batch_size=64 --dataset_root ./dataset/downstream/pets/
# clip_benchmark eval --dataset=flowers --task=zeroshot_classification --pretrained=$pretrained --model=$model --output=$dir/result_flowers.json --batch_size=64 --dataset_root ./dataset/downstream/flowers/
# clip_benchmark eval --dataset=caltech101 --task=zeroshot_classification --pretrained=$pretrained --model=$model --output=$dir/result_caltech101.json --batch_size=64 --dataset_root ./dataset/downstream/caltech101/