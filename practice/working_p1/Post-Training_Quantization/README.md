# Inference time after Quantization

## Unquantized model
==============Summary of All Runs w/ Different Performance Options==============
INFO:           cpu w/ 4 threads: count=50 first=34893 curr=33267 min=30791 max=37476 avg=33904.9 std=1780
INFO: cpu w/ 4 threads (xnnpack): count=50 first=43192 curr=38624 min=33059 max=48052 avg=39220.2 std=2730
INFO: cpu w/ 2 threads (xnnpack): count=50 first=48408 curr=37070 min=35896 max=110280 avg=41921.3 std=14156
INFO:           cpu w/ 2 threads: count=50 first=45762 curr=43296 min=41955 max=46102 avg=44573 std=1199
INFO:                gpu-default: count=50 first=59303 curr=58209 min=56596 max=59303 avg=57705.5 std=705
INFO: cpu w/ 1 threads (xnnpack): count=50 first=62928 curr=62220 min=59119 max=63187 avg=61070 std=1146
INFO:           cpu w/ 1 threads: count=50 first=63146 curr=63514 min=60549 max=65114 avg=62723.4 std=1264

## FP16 Quantization
==============Summary of All Runs w/ Different Performance Options==============
INFO: cpu w/ 4 threads (xnnpack): count=50 first=38276 curr=41256 min=30033 max=41256 avg=34673.5 std=2347
INFO:           cpu w/ 4 threads: count=50 first=37415 curr=39440 min=35977 max=47686 avg=39937.7 std=2597
INFO:           cpu w/ 2 threads: count=50 first=45647 curr=49942 min=45552 max=53329 avg=48602.2 std=1677
INFO: cpu w/ 2 threads (xnnpack): count=50 first=51091 curr=46730 min=44272 max=119454 avg=55170.9 std=14313
INFO: cpu w/ 1 threads (xnnpack): count=50 first=62862 curr=64838 min=60363 max=70098 avg=62946 std=1723
INFO:                gpu-default: count=50 first=65213 curr=62519 min=60344 max=77515 avg=64018.8 std=3051
INFO:           cpu w/ 1 threads: count=50 first=66027 curr=65129 min=62093 max=81266 avg=64855.8 std=2909


## Dynamic Range Quantization
==============Summary of All Runs w/ Different Performance Options==============
INFO: cpu w/ 1 threads (xnnpack): count=50 first=99189 curr=99144 min=95864 max=101700 avg=98936.8 std=1456
INFO: cpu w/ 4 threads (xnnpack): count=50 first=109514 curr=111269 min=95359 max=132744 avg=107331 std=8974
INFO: cpu w/ 2 threads (xnnpack): count=50 first=108754 curr=107077 min=103311 max=129916 avg=107661 std=3890
INFO:           cpu w/ 2 threads: count=50 first=117816 curr=117351 min=110820 max=120431 avg=115431 std=1936
INFO:           cpu w/ 1 threads: count=50 first=120213 curr=118023 min=117219 max=144129 avg=121102 std=3910
INFO:                gpu-default: count=50 first=124576 curr=121429 min=117335 max=127526 avg=121437 std=2323
INFO:           cpu w/ 4 threads: count=50 first=130976 curr=118367 min=115581 max=155916 avg=128087 std=12554

