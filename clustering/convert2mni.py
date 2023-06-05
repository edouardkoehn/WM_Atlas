import clustering.utils as utils
import clustering.utils_nifti as utils_nifti

# load the
ids = [
    101915,
    103414,
    103818,
    899885,
    857263,
    856766,
    105014,
    105115,
    106016,
    108828,
    111312,
    111716,
    113619,
    113922,
    114419,
    115320,
    116524,
    117122,
    118528,
    118730,
    123117,
    123925,
    124422,
    125525,
    126325,
    118932,
    792564,
    127630,
    127933,
    128127,
    128632,
    129028,
    130013,
    130316,
    131217,
    131722,
    133019,
    133928,
    135225,
    135932,
    136833,
    138534,
    139637,
    140925,
    144832,
    146432,
    147737,
    148335,
    148840,
    149337,
    149539,
    149741,
    151223,
    151526,
    151627,
    153025,
    154734,
    156637,
    159340,
    160123,
    161731,
    162733,
    163129,
    176542,
    178950,
    188347,
    189450,
    190031,
    192540,
    196750,
    198451,
    199655,
    201111,
    208226,
    211417,
    211720,
    212318,
    214423,
    221319,
    239944,
    245333,
    280739,
    298051,
    366446,
    397760,
    414229,
    499566,
    654754,
    672756,
    751348,
    756055,
]

for subject_id in ids:

    f_s = utils.get_nifti_path(subject_id, "rw", "acpc", 2.0)
    f_d = utils.get_transformation_file(subject_id, "acpc2nmi")
    f_r = utils.get_reference_file(subject_id)
    mask = utils.get_mask_path("95")
    path_nifti_in = utils.get_src_nifti(subject_id)

    f_o = f_s[0:-12] + "_extraction_mni.nii.gz"
    f_o_reslice = f_s[0:-12] + "_extraction_mni_reslice.nii.gz"
    path_nifti_original_mni_out = f_s[0:-12] + "_original_mni.nii.gz"
    path_nifti_original_acpc_out = f_s[0:-12] + "_original_acpc.nii"

    utils_nifti.hb_nii_displace(f_s, f_d, f_r, f_o)
    utils_nifti.hb_reslice_vol(f_o, mask, f_o_reslice)
    utils_nifti.copy_nifti(
        path_nifti_in,
        path_nifti_original_acpc_out,
        type="mni",
        f_s=path_nifti_in,
        f_d=f_d,
        f_r=f_r,
        f_o=path_nifti_original_mni_out,
    )
