#!/bin/bash
nifity_type="reslice"
save="true"
value_type=( "distance" "cluster" "z_score" )

method="sym"
k_eigen="12"
cluster="10"

# ids_0=" -i 131217 -i 103414 -i 139637 -i 214423 -i 136833 -i 159340 -i 857263 -i 149337 -i 115320 -i 414229 -i 199655 -i 756055 -i 130013 -i 151526 -i 147737 -i 117122 -i 148840 -i 101309 -i 154734 -i 161731 -i 116524 -i 198451 -i 118932 -i 128127 -i 792564"
# ids_1=" -i 212318 -i 148840 -i 133019 -i 113619 -i 111312 -i 189450 -i 101309 -i 153025 -i 178950 -i 151627 -i 140925 -i 149539 -i 161731 -i 136833 -i 149337 -i 133928 -i 129028 -i 149741 -i 127630 -i 123925 -i 151223 -i 159340 -i 113922 -i 751348 -i 108828"
# ids_2=" -i 101915 -i 116524 -i 136833 -i 196750 -i 857263 -i 672756 -i 144832 -i 113922 -i 140925 -i 111312 -i 298051 -i 162733 -i 129028 -i 499566 -i 198451 -i 188347 -i 899885 -i 151526 -i 123117 -i 103414 -i 113619 -i 130316 -i 103818 -i 118730 -i 154734"
# ids_3=" -i 105014 -i 118932 -i 108828 -i 118528 -i 111312 -i 130316 -i 124422 -i 245333 -i 163129 -i 160123 -i 397760 -i 221319 -i 211417 -i 125525 -i 176542 -i 131722 -i 190031 -i 239944 -i 103414 -i 189450 -i 414229 -i 131217 -i 149337 -i 118730 -i 899885"
# ids_4=" -i 106016 -i 148335 -i 499566 -i 857263 -i 163129 -i 151526 -i 192540 -i 131722 -i 208226 -i 135932 -i 103414 -i 190031 -i 211720 -i 123925 -i 144832 -i 756055 -i 856766 -i 899885 -i 188347 -i 127630 -i 161731 -i 280739 -i 151223 -i 154734 -i 130316"
# ids_5=" -i 792564 -i 163129 -i 156637 -i 106016 -i 131217 -i 115320 -i 176542 -i 124422 -i 245333 -i 298051 -i 133928 -i 105014 -i 130013 -i 138534 -i 198451 -i 133019 -i 159340 -i 130316 -i 118932 -i 756055 -i 139637 -i 208226 -i 113619 -i 108828 -i 144832"
# ids_6=" -i 153025 -i 239944 -i 105115 -i 672756 -i 129028 -i 123117 -i 178950 -i 105014 -i 298051 -i 212318 -i 106016 -i 115320 -i 148840 -i 899885 -i 856766 -i 103414 -i 211417 -i 792564 -i 280739 -i 214423 -i 101309 -i 189450 -i 128632 -i 130316 -i 199655"
# ids_7=" -i 280739 -i 149741 -i 115320 -i 198451 -i 153025 -i 756055 -i 151526 -i 113922 -i 140925 -i 221319 -i 899885 -i 116524 -i 108828 -i 208226 -i 188347 -i 151223 -i 133019 -i 118730 -i 111716 -i 129028 -i 131722 -i 159340 -i 397760 -i 128127 -i 136833"
# ids_8=" -i 151223 -i 153025 -i 129028 -i 189450 -i 245333 -i 151627 -i 139637 -i 108828 -i 149539 -i 159340 -i 135932 -i 154734 -i 136833 -i 127630 -i 192540 -i 298051 -i 212318 -i 130013 -i 151526 -i 211720 -i 756055 -i 414229 -i 161731 -i 106016 -i 214423"
# ids_9=" -i 101915 -i 116524 -i 117122 -i 672756 -i 153025 -i 135932 -i 146432 -i 118932 -i 123925 -i 178950 -i 211720 -i 163129 -i 751348 -i 139637 -i 124422 -i 159340 -i 118528 -i 113619 -i 128127 -i 127630 -i 160123 -i 128632 -i 857263 -i 198451 -i 129028"

ids_all=" -i 101309 -i 101915 -i 103414 -i 149539 -i 116524 -i 123117 -i 135932 -i 160123 -i 176542 -i 147737 -i 111716 -i 298051 -i 123925 -i 118528 -i 131722 -i 139637 -i 106016 -i 154734 -i 856766 -i 136833 -i 201111 -i 103818 -i 208226 -i 140925 -i 792564 -i 189450 -i 130013 -i 138534 -i 899885 -i 118730 -i 178950 -i 751348 -i 245333 -i 280739 -i 130316 -i 156637 -i 149337 -i 211720 -i 108828 -i 199655 -i 105115 -i 146432 -i 192540 -i 397760 -i 129028 -i 144832 -i 756055 -i 135225 -i 499566 -i 124422 -i 159340 -i 126325 -i 114419 -i 654754 -i 188347 -i 162733 -i 133928 -i 118932 -i 133019 -i 414229 -i 148335 -i 151223 -i 161731 -i 672756 -i 214423 -i 128127 -i 211417 -i 105014 -i 366446 -i 857263 -i 125525 -i 198451 -i 239944 -i 111312 -i 131217 -i 128632 -i 149741 -i 117122 -i 153025 -i 127630 -i 113922 -i 151526 -i 196750 -i 212318 -i 148840 -i 190031 -i 163129 -i 113619 -i 151627 -i 221319 -i 115320 -i 127933"

# echo clustering_boostrap $ids_0 -m $method -k $k_eigen -n $nifity_type -c $cluster -s $save -v ${value_type[0]} -v ${value_type[1]} -v ${value_type[2]} -b 1 &
# clustering_boostrap $ids_0 -m $method -k $k_eigen -n $nifity_type -c $cluster -s $save -v ${value_type[0]} -v ${value_type[1]} -v ${value_type[2]} -b 1 &

# echo clustering_boostrap $ids_1 -m $method -k $k_eigen -n $nifity_type -c $cluster -s $save -v ${value_type[0]} -v ${value_type[1]} -v ${value_type[2]} -b 2 &
# clustering_boostrap $ids_1 -m $method -k $k_eigen -n $nifity_type -c $cluster -s $save -v ${value_type[0]} -v ${value_type[1]} -v ${value_type[2]} -b 2 &

# echo clustering_boostrap $ids_2 -m $method -k $k_eigen -n $nifity_type -c $cluster -s $save -v ${value_type[0]} -v ${value_type[1]} -v ${value_type[2]} -b 3 &
# clustering_boostrap $ids_2 -m $method -k $k_eigen -n $nifity_type -c $cluster -s $save -v ${value_type[0]} -v ${value_type[1]} -v ${value_type[2]} -b 3 &

# echo clustering_boostrap $ids_3 -m $method -k $k_eigen -n $nifity_type -c $cluster -s $save -v ${value_type[0]} -v ${value_type[1]} -v ${value_type[2]} -b 4 &
# clustering_boostrap $ids_3 -m $method -k $k_eigen -n $nifity_type -c $cluster -s $save -v ${value_type[0]} -v ${value_type[1]} -v ${value_type[2]} -b 4 &

# echo clustering_boostrap $ids_4 -m $method -k $k_eigen -n $nifity_type -c $cluster -s $save -v ${value_type[0]} -v ${value_type[1]} -v ${value_type[2]} -b 5 &
# clustering_boostrap $ids_4 -m $method -k $k_eigen -n $nifity_type -c $cluster -s $save -v ${value_type[0]} -v ${value_type[1]} -v ${value_type[2]} -b 5 &

# echo clustering_boostrap $ids_5 -m $method -k $k_eigen -n $nifity_type -c $cluster -s $save -v ${value_type[0]} -v ${value_type[1]} -v ${value_type[2]} -b 6 &
# clustering_boostrap $ids_5 -m $method -k $k_eigen -n $nifity_type -c $cluster -s $save -v ${value_type[0]} -v ${value_type[1]} -v ${value_type[2]} -b 6 &

# echo clustering_boostrap $ids_6 -m $method -k $k_eigen -n $nifity_type -c $cluster -s $save -v ${value_type[0]} -v ${value_type[1]} -v ${value_type[2]} -b 7 &
# clustering_boostrap $ids_6 -m $method -k $k_eigen -n $nifity_type -c $cluster -s $save -v ${value_type[0]} -v ${value_type[1]} -v ${value_type[2]} -b 7 &

# echo clustering_boostrap $ids_7 -m $method -k $k_eigen -n $nifity_type -c $cluster -s $save -v ${value_type[0]} -v ${value_type[1]} -v ${value_type[2]} -b 8 &
# clustering_boostrap $ids_7 -m $method -k $k_eigen -n $nifity_type -c $cluster -s $save -v ${value_type[0]} -v ${value_type[1]} -v ${value_type[2]} -b 8 &

# echo clustering_boostrap $ids_8 -m $method -k $k_eigen -n $nifity_type -c $cluster -s $save -v ${value_type[0]} -v ${value_type[1]} -v ${value_type[2]} -b 9 &
# clustering_boostrap $ids_8 -m $method -k $k_eigen -n $nifity_type -c $cluster -s $save -v ${value_type[0]} -v ${value_type[1]} -v ${value_type[2]} -b 9 &

# echo clustering_boostrap $ids_9 -m $method -k $k_eigen -n $nifity_type -c $cluster -s $save -v ${value_type[0]} -v ${value_type[1]} -v ${value_type[2]} -b 10 &
# clustering_boostrap $ids_9 -m $method -k $k_eigen -n $nifity_type -c $cluster -s $save -v ${value_type[0]} -v ${value_type[1]} -v ${value_type[2]} -b 10 &

echo clustering_boostrap $ids_all -m $method -k $k_eigen -n $nifity_type -c $cluster -s $save -v ${value_type[0]} -v ${value_type[1]} -v ${value_type[2]} -b 10 &
clustering_boostrap $ids_all -m $method -k $k_eigen -n $nifity_type -c $cluster -s $save -v ${value_type[0]} -v ${value_type[1]} -v ${value_type[2]} -b 10 &
