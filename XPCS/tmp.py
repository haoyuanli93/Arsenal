import numpy as np

##############################################################
#     Both beams
##############################################################
kBarBin_01 = kbarPerQ_both[:, 0] * pixel_num_per_q[0]
kBarBin_01 += kbarPerQ_both[:, 1] * pixel_num_per_q[1]
kBarBin_01 /= (pixel_num_per_q[0] + pixel_num_per_q[1])

kBarBin_234 = kbarPerQ_both[:, 2] * pixel_num_per_q[2]
kBarBin_234 += kbarPerQ_both[:, 3] * pixel_num_per_q[3]
kBarBin_234 += kbarPerQ_both[:, 4] * pixel_num_per_q[4]
kBarBin_234 /= (pixel_num_per_q[2] + pixel_num_per_q[3] + pixel_num_per_q[4])

probability_01 = probabilityPerQ_with0_both[:, 0] * pixel_num_per_q[0]
probability_01 += probabilityPerQ_with0_both[:, 1] * pixel_num_per_q[1]
probability_01 /= (pixel_num_per_q[0] + pixel_num_per_q[1])

probability_234 = probabilityPerQ_with0_both[:, 2] * pixel_num_per_q[2]
probability_234 += probabilityPerQ_with0_both[:, 3] * pixel_num_per_q[3]
probability_234 += probabilityPerQ_with0_both[:, 4] * pixel_num_per_q[4]
probability_234 /= (pixel_num_per_q[2] + pixel_num_per_q[3] + pixel_num_per_q[4])

kBarBin_01_mask = np.ones_like(kBarBin_01, dtype=bool)
kBarBin_01_mask[kBarBin_01 < 1e-6] = False

kBarBin_234_mask = np.ones_like(kBarBin_234, dtype=bool)
kBarBin_234_mask[kBarBin_234 < 1e-6] = False

# Save the data
np.savez("./data/kbar_both_1ps.npz",
         kBarQ1=kBarBin_01,
         kBarMaskQ1=kBarBin_01_mask,
         kBarQ2=kBarBin_234,
         kBarMaskQ2=kBarBin_234_mask)

np.savez("./data/probability_both_1ps.npz",
         ProbabilityQ1=probability_01,
         ProbabilityQ2=probability_234, )

##############################################################
#     CC
##############################################################
kBarBin_01 = kbarPerQ_cc[:, 0] * pixel_num_per_q[0]
kBarBin_01 += kbarPerQ_cc[:, 1] * pixel_num_per_q[1]
kBarBin_01 /= (pixel_num_per_q[0] + pixel_num_per_q[1])

kBarBin_234 = kbarPerQ_cc[:, 2] * pixel_num_per_q[2]
kBarBin_234 += kbarPerQ_cc[:, 3] * pixel_num_per_q[3]
kBarBin_234 += kbarPerQ_cc[:, 4] * pixel_num_per_q[4]
kBarBin_234 /= (pixel_num_per_q[2] + pixel_num_per_q[3] + pixel_num_per_q[4])

probability_01 = probabilityPerQ_with0_cc[:, 0] * pixel_num_per_q[0]
probability_01 += probabilityPerQ_with0_cc[:, 1] * pixel_num_per_q[1]
probability_01 /= (pixel_num_per_q[0] + pixel_num_per_q[1])

probability_234 = probabilityPerQ_with0_cc[:, 2] * pixel_num_per_q[2]
probability_234 += probabilityPerQ_with0_cc[:, 3] * pixel_num_per_q[3]
probability_234 += probabilityPerQ_with0_cc[:, 4] * pixel_num_per_q[4]
probability_234 /= (pixel_num_per_q[2] + pixel_num_per_q[3] + pixel_num_per_q[4])

kBarBin_01_mask = np.ones_like(kBarBin_01, dtype=bool)
kBarBin_01_mask[kBarBin_01 < 1e-6] = False

kBarBin_234_mask = np.ones_like(kBarBin_234, dtype=bool)
kBarBin_234_mask[kBarBin_234 < 1e-6] = False

# Save the data
np.savez("./data/kbar_cc_1ps.npz",
         kBarQ1=kBarBin_01,
         kBarMaskQ1=kBarBin_01_mask,
         kBarQ2=kBarBin_234,
         kBarMaskQ2=kBarBin_234_mask)

np.savez("./data/probability_cc_1ps.npz",
         ProbabilityQ1=probability_01,
         ProbabilityQ2=probability_234, )

##############################################################
#     VCC
##############################################################
kBarBin_01 = kbarPerQ_vcc[:, 0] * pixel_num_per_q[0]
kBarBin_01 += kbarPerQ_vcc[:, 1] * pixel_num_per_q[1]
kBarBin_01 /= (pixel_num_per_q[0] + pixel_num_per_q[1])

kBarBin_234 = kbarPerQ_vcc[:, 2] * pixel_num_per_q[2]
kBarBin_234 += kbarPerQ_vcc[:, 3] * pixel_num_per_q[3]
kBarBin_234 += kbarPerQ_vcc[:, 4] * pixel_num_per_q[4]
kBarBin_234 /= (pixel_num_per_q[2] + pixel_num_per_q[3] + pixel_num_per_q[4])

probability_01 = probabilityPerQ_with0_vcc[:, 0] * pixel_num_per_q[0]
probability_01 += probabilityPerQ_with0_vcc[:, 1] * pixel_num_per_q[1]
probability_01 /= (pixel_num_per_q[0] + pixel_num_per_q[1])

probability_234 = probabilityPerQ_with0_vcc[:, 2] * pixel_num_per_q[2]
probability_234 += probabilityPerQ_with0_vcc[:, 3] * pixel_num_per_q[3]
probability_234 += probabilityPerQ_with0_vcc[:, 4] * pixel_num_per_q[4]
probability_234 /= (pixel_num_per_q[2] + pixel_num_per_q[3] + pixel_num_per_q[4])

kBarBin_01_mask = np.ones_like(kBarBin_01, dtype=bool)
kBarBin_01_mask[kBarBin_01 < 1e-6] = False

kBarBin_234_mask = np.ones_like(kBarBin_234, dtype=bool)
kBarBin_234_mask[kBarBin_234 < 1e-6] = False

# Save the data
np.savez("./data/kbar_vcc_1ps.npz",
         kBarQ1=kBarBin_01,
         kBarMaskQ1=kBarBin_01_mask,
         kBarQ2=kBarBin_234,
         kBarMaskQ2=kBarBin_234_mask)

np.savez("./data/probability_vcc_1ps.npz",
         ProbabilityQ1=probability_01,
         ProbabilityQ2=probability_234, )

##############################################################################
#   Save data for the 7ps case
##############################################################################
##############################################################
#     Both beams
##############################################################
kBarBin_01 = kbarPerQ_both[:, 0] * pixel_num_per_q[0]
kBarBin_01 += kbarPerQ_both[:, 1] * pixel_num_per_q[1]
kBarBin_01 /= (pixel_num_per_q[0] + pixel_num_per_q[1])

kBarBin_234 = kbarPerQ_both[:, 2] * pixel_num_per_q[2]
kBarBin_234 += kbarPerQ_both[:, 3] * pixel_num_per_q[3]
kBarBin_234 += kbarPerQ_both[:, 4] * pixel_num_per_q[4]
kBarBin_234 /= (pixel_num_per_q[2] + pixel_num_per_q[3] + pixel_num_per_q[4])

probability_01 = probabilityPerQ_with0_both[:, 0] * pixel_num_per_q[0]
probability_01 += probabilityPerQ_with0_both[:, 1] * pixel_num_per_q[1]
probability_01 /= (pixel_num_per_q[0] + pixel_num_per_q[1])

probability_234 = probabilityPerQ_with0_both[:, 2] * pixel_num_per_q[2]
probability_234 += probabilityPerQ_with0_both[:, 3] * pixel_num_per_q[3]
probability_234 += probabilityPerQ_with0_both[:, 4] * pixel_num_per_q[4]
probability_234 /= (pixel_num_per_q[2] + pixel_num_per_q[3] + pixel_num_per_q[4])

kBarBin_01_mask = np.ones_like(kBarBin_01, dtype=bool)
kBarBin_01_mask[kBarBin_01 < 1e-6] = False

kBarBin_234_mask = np.ones_like(kBarBin_234, dtype=bool)
kBarBin_234_mask[kBarBin_234 < 1e-6] = False

# Save the data
np.savez("./data/kbar_both_7ps.npz",
         kBarQ1=kBarBin_01,
         kBarMaskQ1=kBarBin_01_mask,
         kBarQ2=kBarBin_234,
         kBarMaskQ2=kBarBin_234_mask)

np.savez("./data/probability_both_7ps.npz",
         ProbabilityQ1=probability_01,
         ProbabilityQ2=probability_234, )

##############################################################
#     CC
##############################################################
kBarBin_01 = kbarPerQ_cc[:, 0] * pixel_num_per_q[0]
kBarBin_01 += kbarPerQ_cc[:, 1] * pixel_num_per_q[1]
kBarBin_01 /= (pixel_num_per_q[0] + pixel_num_per_q[1])

kBarBin_234 = kbarPerQ_cc[:, 2] * pixel_num_per_q[2]
kBarBin_234 += kbarPerQ_cc[:, 3] * pixel_num_per_q[3]
kBarBin_234 += kbarPerQ_cc[:, 4] * pixel_num_per_q[4]
kBarBin_234 /= (pixel_num_per_q[2] + pixel_num_per_q[3] + pixel_num_per_q[4])

probability_01 = probabilityPerQ_with0_cc[:, 0] * pixel_num_per_q[0]
probability_01 += probabilityPerQ_with0_cc[:, 1] * pixel_num_per_q[1]
probability_01 /= (pixel_num_per_q[0] + pixel_num_per_q[1])

probability_234 = probabilityPerQ_with0_cc[:, 2] * pixel_num_per_q[2]
probability_234 += probabilityPerQ_with0_cc[:, 3] * pixel_num_per_q[3]
probability_234 += probabilityPerQ_with0_cc[:, 4] * pixel_num_per_q[4]
probability_234 /= (pixel_num_per_q[2] + pixel_num_per_q[3] + pixel_num_per_q[4])

kBarBin_01_mask = np.ones_like(kBarBin_01, dtype=bool)
kBarBin_01_mask[kBarBin_01 < 1e-6] = False

kBarBin_234_mask = np.ones_like(kBarBin_234, dtype=bool)
kBarBin_234_mask[kBarBin_234 < 1e-6] = False

# Save the data
np.savez("./data/kbar_cc_7ps.npz",
         kBarQ1=kBarBin_01,
         kBarMaskQ1=kBarBin_01_mask,
         kBarQ2=kBarBin_234,
         kBarMaskQ2=kBarBin_234_mask)

np.savez("./data/probability_cc_7ps.npz",
         ProbabilityQ1=probability_01,
         ProbabilityQ2=probability_234, )

##############################################################
#     VCC
##############################################################
kBarBin_01 = kbarPerQ_vcc[:, 0] * pixel_num_per_q[0]
kBarBin_01 += kbarPerQ_vcc[:, 1] * pixel_num_per_q[1]
kBarBin_01 /= (pixel_num_per_q[0] + pixel_num_per_q[1])

kBarBin_234 = kbarPerQ_vcc[:, 2] * pixel_num_per_q[2]
kBarBin_234 += kbarPerQ_vcc[:, 3] * pixel_num_per_q[3]
kBarBin_234 += kbarPerQ_vcc[:, 4] * pixel_num_per_q[4]
kBarBin_234 /= (pixel_num_per_q[2] + pixel_num_per_q[3] + pixel_num_per_q[4])

probability_01 = probabilityPerQ_with0_vcc[:, 0] * pixel_num_per_q[0]
probability_01 += probabilityPerQ_with0_vcc[:, 1] * pixel_num_per_q[1]
probability_01 /= (pixel_num_per_q[0] + pixel_num_per_q[1])

probability_234 = probabilityPerQ_with0_vcc[:, 2] * pixel_num_per_q[2]
probability_234 += probabilityPerQ_with0_vcc[:, 3] * pixel_num_per_q[3]
probability_234 += probabilityPerQ_with0_vcc[:, 4] * pixel_num_per_q[4]
probability_234 /= (pixel_num_per_q[2] + pixel_num_per_q[3] + pixel_num_per_q[4])

kBarBin_01_mask = np.ones_like(kBarBin_01, dtype=bool)
kBarBin_01_mask[kBarBin_01 < 1e-6] = False

kBarBin_234_mask = np.ones_like(kBarBin_234, dtype=bool)
kBarBin_234_mask[kBarBin_234 < 1e-6] = False

# Save the data
np.savez("./data/kbar_vcc_7ps.npz",
         kBarQ1=kBarBin_01,
         kBarMaskQ1=kBarBin_01_mask,
         kBarQ2=kBarBin_234,
         kBarMaskQ2=kBarBin_234_mask)

np.savez("./data/probability_vcc_7ps.npz",
         ProbabilityQ1=probability_01,
         ProbabilityQ2=probability_234, )

###################################################################################
#    Load data
###################################################################################
waterData_1ps = {"Q1": {"cc": {"kbar": np.load("../Analysis_water_v2/data/kbar_cc_1ps.npz")['kBarQ1'][:],
                               "probability": np.load("../Analysis_water_v2/data/probability_cc_1ps.npz")[
                                                  'ProbabilityQ1'][:],
                               "kbarMask": np.load("../Analysis_water_v2/data/kbar_cc_1ps.npz")['kBarMaskQ1'][:],
                               },
                        "vcc": {"kbar": np.load("../Analysis_water_v2/data/kbar_vcc_1ps.npz")['kBarQ1'][:],
                                "probability": np.load("../Analysis_water_v2/data/probability_vcc_1ps.npz")[
                                                   'ProbabilityQ1'][:],
                                "kbarMask": np.load("../Analysis_water_v2/data/kbar_vcc_1ps.npz")['kBarMaskQ1'][:],
                                },
                        "both": {"kbar": np.load("../Analysis_water_v2/data/kbar_both_1ps.npz")['kBarQ1'][:],
                                 "probability": np.load("../Analysis_water_v2/data/probability_both_1ps.npz")[
                                                    'ProbabilityQ1'][:],
                                 "kbarMask": np.load("../Analysis_water_v2/data/kbar_both_1ps.npz")['kBarMaskQ1'][:],
                                 },
                        },
                 "Q2": {"cc": {"kbar": np.load("../Analysis_water_v2/data/kbar_cc_1ps.npz")['kBarQ2'][:],
                               "probability": np.load("../Analysis_water_v2/data/probability_cc_1ps.npz")[
                                                  'ProbabilityQ2'][:],
                               "kbarMask": np.load("../Analysis_water_v2/data/kbar_cc_1ps.npz")['kBarMaskQ2'][:],
                               },
                        "vcc": {"kbar": np.load("../Analysis_water_v2/data/kbar_vcc_1ps.npz")['kBarQ2'][:],
                                "probability": np.load("../Analysis_water_v2/data/probability_vcc_1ps.npz")[
                                                   'ProbabilityQ2'][:],
                                "kbarMask": np.load("../Analysis_water_v2/data/kbar_vcc_1ps.npz")['kBarMaskQ2'][:],
                                },
                        "both": {"kbar": np.load("../Analysis_water_v2/data/kbar_both_1ps.npz")['kBarQ2'][:],
                                 "probability": np.load("../Analysis_water_v2/data/probability_both_1ps.npz")[
                                                    'ProbabilityQ2'][:],
                                 "kbarMask": np.load("../Analysis_water_v2/data/kbar_both_1ps.npz")['kBarMaskQ2'][:],
                                 },
                        },
                 }

waterData_7ps = {"Q1": {"cc": {"kbar": np.load("../Analysis_water_v2/data/kbar_cc_7ps.npz")['kBarQ1'][:],
                               "probability": np.load("../Analysis_water_v2/data/probability_cc_7ps.npz")[
                                                  'ProbabilityQ1'][:],
                               "kbarMask": np.load("../Analysis_water_v2/data/kbar_cc_7ps.npz")['kBarMaskQ1'][:],
                               },
                        "vcc": {"kbar": np.load("../Analysis_water_v2/data/kbar_vcc_7ps.npz")['kBarQ1'][:],
                                "probability": np.load("../Analysis_water_v2/data/probability_vcc_7ps.npz")[
                                                   'ProbabilityQ1'][:],
                                "kbarMask": np.load("../Analysis_water_v2/data/kbar_vcc_7ps.npz")['kBarMaskQ1'][:],
                                },
                        "both": {"kbar": np.load("../Analysis_water_v2/data/kbar_both_7ps.npz")['kBarQ1'][:],
                                 "probability": np.load("../Analysis_water_v2/data/probability_both_7ps.npz")[
                                                    'ProbabilityQ1'][:],
                                 "kbarMask": np.load("../Analysis_water_v2/data/kbar_both_7ps.npz")['kBarMaskQ1'][:],
                                 },
                        },
                 "Q2": {"cc": {"kbar": np.load("../Analysis_water_v2/data/kbar_cc_7ps.npz")['kBarQ2'][:],
                               "probability": np.load("../Analysis_water_v2/data/probability_cc_7ps.npz")[
                                                  'ProbabilityQ2'][:],
                               "kbarMask": np.load("../Analysis_water_v2/data/kbar_cc_7ps.npz")['kBarMaskQ2'][:],
                               },
                        "vcc": {"kbar": np.load("../Analysis_water_v2/data/kbar_vcc_7ps.npz")['kBarQ2'][:],
                                "probability": np.load("../Analysis_water_v2/data/probability_vcc_7ps.npz")[
                                                   'ProbabilityQ2'][:],
                                "kbarMask": np.load("../Analysis_water_v2/data/kbar_vcc_7ps.npz")['kBarMaskQ2'][:],
                                },
                        "both": {"kbar": np.load("../Analysis_water_v2/data/kbar_both_7ps.npz")['kBarQ2'][:],
                                 "probability": np.load("../Analysis_water_v2/data/probability_both_7ps.npz")[
                                                    'ProbabilityQ2'][:],
                                 "kbarMask": np.load("../Analysis_water_v2/data/kbar_both_7ps.npz")['kBarMaskQ2'][:],
                                 },
                        },
                 }

#########################################################################
#########################################################################
#
#        Get the intermediate scattering function
#
#########################################################################
#########################################################################
waterContrast_1ps = [XPCSutil.getISF(probability_CC=waterData_1ps['Q1']['cc']['probability'][:, :4][waterData_1ps['Q1']['cc']['kbarMask']],
                                     kMean_CC=waterData_1ps['Q1']['cc']['kbar'][waterData_1ps['Q1']['cc']['kbarMask']],
                                     probability_VCC=waterData_1ps['Q1']['vcc']['probability'][:, :4][waterData_1ps['Q1']['vcc']['kbarMask']],
                                     kMean_VCC=waterData_1ps['Q1']['vcc']['kbar'][waterData_1ps['Q1']['vcc']['kbarMask']],
                                     probability_Both=waterData_1ps['Q1']['both']['probability'][:, :4][waterData_1ps['Q1']['both']['kbarMask']],
                                     kMean_Both=waterData_1ps['Q1']['both']['kbar'][waterData_1ps['Q1']['both']['kbarMask']],
                                     nroi=687383.0,
                                     effectiveOverlap=mu_1ps),
                     XPCSutil.getISF(probability_CC=waterData_1ps['Q2']['cc']['probability'][:, :4][waterData_1ps['Q2']['cc']['kbarMask']],
                                     kMean_CC=waterData_1ps['Q2']['cc']['kbar'][waterData_1ps['Q2']['cc']['kbarMask']],
                                     probability_VCC=waterData_1ps['Q2']['vcc']['probability'][:, :4][waterData_1ps['Q2']['vcc']['kbarMask']],
                                     kMean_VCC=waterData_1ps['Q2']['vcc']['kbar'][waterData_1ps['Q2']['vcc']['kbarMask']],
                                     probability_Both=waterData_1ps['Q2']['both']['probability'][:, :4][waterData_1ps['Q2']['both']['kbarMask']],
                                     kMean_Both=waterData_1ps['Q2']['both']['kbar'][waterData_1ps['Q2']['both']['kbarMask']],
                                     nroi=1072803.0,
                                     effectiveOverlap=mu_1ps),
                     ]

print("================================================================")
print("The ISF 1ps is {:.2f} for Q1 and {:.2f} for Q2".format(waterContrast_1ps[0][0], waterContrast_1ps[1][0]))
print("================================================================")

waterContrast_7ps = [XPCSutil.getISF(probability_CC=waterData_7ps['Q1']['cc']['probability'][:, :4][waterData_7ps['Q1']['cc']['kbarMask']],
                                     kMean_CC=waterData_7ps['Q1']['cc']['kbar'][waterData_7ps['Q1']['cc']['kbarMask']],
                                     probability_VCC=waterData_7ps['Q1']['vcc']['probability'][:, :4][waterData_7ps['Q1']['vcc']['kbarMask']],
                                     kMean_VCC=waterData_7ps['Q1']['vcc']['kbar'][waterData_7ps['Q1']['vcc']['kbarMask']],
                                     probability_Both=waterData_7ps['Q1']['both']['probability'][:, :4][waterData_7ps['Q1']['both']['kbarMask']],
                                     kMean_Both=waterData_7ps['Q1']['both']['kbar'][waterData_7ps['Q1']['both']['kbarMask']],
                                     nroi=687383.0,
                                     effectiveOverlap=mu_7ps),
                     XPCSutil.getISF(probability_CC=waterData_7ps['Q2']['cc']['probability'][:, :4][waterData_7ps['Q2']['cc']['kbarMask']],
                                     kMean_CC=waterData_7ps['Q2']['cc']['kbar'][waterData_7ps['Q2']['cc']['kbarMask']],
                                     probability_VCC=waterData_7ps['Q2']['vcc']['probability'][:, :4][waterData_7ps['Q2']['vcc']['kbarMask']],
                                     kMean_VCC=waterData_7ps['Q2']['vcc']['kbar'][waterData_7ps['Q2']['vcc']['kbarMask']],
                                     probability_Both=waterData_7ps['Q2']['both']['probability'][:, :4][waterData_7ps['Q2']['both']['kbarMask']],
                                     kMean_Both=waterData_7ps['Q2']['both']['kbar'][waterData_7ps['Q2']['both']['kbarMask']],
                                     nroi=1072803.0,
                                     effectiveOverlap=mu_7ps),
                     ]

print("================================================================")
print("The ISF 7ps is {:.2f} for Q1 and {:.2f} for Q2".format(waterContrast_7ps[0][0], waterContrast_7ps[1][0]))
print("================================================================")
