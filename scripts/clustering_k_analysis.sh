#!/bin/bash
method="rw"
k_eigen="100"
nifity_type="reslice"
save="true"
value_type=( "distance" "cluster" "z_score" )

ids=(  100307  103111  105115  100408  103414  106016
        101107  103818  110411  101915 )

ks=( 2 10 20 30 40 50 60 70 80 90 100)

# ids=(101309)
stirng=""
for i in "${ids[@]}"

do
    string="$string -i $i"

done

for k in  "${ks[@]}"
do
    echo clustering_pop $string -m $method -k $k -n $nifity_type -s $save -v ${value_type[0]} -v ${value_type[1]} -v ${value_type[2]} &
    clustering_pop $string -m $method -k $k -n $nifity_type -s $save -v ${value_type[0]} -v ${value_type[1]} -v ${value_type[2]} &
done
