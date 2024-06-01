import subprocess


imgs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  
vals = [1, 1, 2, 2, 2, 3, 3, 3, 4, 4]  
seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] 
mahalanobis_values = ["normal", "regularization"] 
dimension_values = [4, 8, 16, 32, 64]
dim_reduction = ["none", "svd"] 

for img, val in zip(imgs, vals):
    for seed in seeds:
        for mahalanobis in mahalanobis_values:
            for red in dim_reduction:
                if red == "none":
                    dimension_values = [0]
                
                for dim in dimension_values:

                    command = f"python methods.py \
                        --root \"./pineapples_5m\" \
                        --num-classes 0 \
                        --load-pretrained \
                        --timm-model \"xcit_nano_12_p8_224.fb_dist_in1k\" \
                        --loss \"mae\" \
                        --optim \"sgd\" \
                        --val-freq \"1\" \
                        --use-semi-split \
                        --semi-percentage \"1.0\" \
                        --dataset \"coco2017\" \
                        --batch-size \"4\" \
                        --batch-size-val \"1\" \
                        --batch-size-labeled \"10\" \
                        --batch-size-unlabeled \"10\" \
                        --batch-size-validation \"1\" \
                        --ood-labeled-samples \"{img}\" --ood-unlabeled-samples \"100\" --ood-validation-samples \"{val}\" \
                        --aug-method \"rand_augmentation\" \
                        --img-resolution \"1024\" \
                        --new-sample-size \"256\" \
                        --numa \"-1\" \
                        --output-folder \"PINEAPPLES2_5mts_nosplits\" \
                        --run-name \"temp\" \
                        --seed \"{seed}\" \
                        --sam-model \"h\" \
                        --ood-thresh \"0.8\" \
                        --ood-histogram-bins \"15\" --use-sam-embeddings \"0\" \
                        --method \"fewshotMahalanobis\" \
                        --device \"cuda\" \
                        --sam-proposal \"fastsam\" \ 
                        --dim-red \"{red}\" \
                        --n-components \"{dim}\" \
                        --mahalanobis \"{mahalanobis}\" \
                        --beta \"0\""


                    print("-------------------------------------------------")
                    print(command)
                    # Run the command using subprocess
                    try:
                        # Run the command using subprocess
                        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
                        print(f"Command output (stdout): {e.stdout}")
                    except subprocess.CalledProcessError as e:
                        print(f"Error occurred: {e}")
                        print(f"Command output (stdout): {e.stdout}")
                        print(f"Command error output (stderr): {e.stderr}")