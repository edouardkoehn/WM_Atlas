#!/bin/bash
method="sym"
k_eigen="10"
nifity_type="reslice"
save="true"
value_type=( "distance" "cluster" "z_score" )
#Complete population
ids=(  101309 101915 103414 103818 899885 857263 856766
        105014 105115 106016 108828 111312 111716 113619
        113922 114419 115320 116524 117122 118528 118730
        123117 123925 124422 125525 126325 118932 792564
        127630 127933 128127 128632 129028 130013 130316 131217
        131722 133019 133928 135225 135932 136833 138534 139637
        140925 144832 146432 147737 148335 148840 149337 149539
        149741 151223 151526 151627 153025 154734 156637 159340
        160123 161731 162733 163129 176542 178950 188347 189450
        190031 192540 196750 198451 199655 201111 208226 211417
        211720 212318 214423 221319 239944 245333 280739 298051
        366446 397760 414229 499566 654754 672756 751348 756055)
#5 subject
#ids=( 101309 101915 103414 149539 116524)
#10 subjects
#ids=( 101309 101915 103414 103818 105014 105115 106016 108828 111312 111716 )

stirng=""
for i in "${ids[@]}"

do
    string="$string -i $i"

done


echo clustering_pop $string -m $method -k $k_eigen -n $nifity_type -s $save -v ${value_type[0]} -v ${value_type[1]} -v ${value_type[2]} &
clustering_pop $string -m $method -k $k_eigen -n $nifity_type -s $save -v ${value_type[0]} -v ${value_type[1]} -v ${value_type[2]} &