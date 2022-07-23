waterContrast_1ps = [XPCSutil.getISF(probability_CC=waterData_1ps['Q1']['cc']['probability'][:, :4][waterData_1ps['Q1']['cc']['kbarMask']][80000:160000],
                                     kMean_CC=waterData_1ps['Q1']['cc']['kbar'][waterData_1ps['Q1']['cc']['kbarMask']][80000:160000],
                                     probability_VCC=waterData_1ps['Q1']['vcc']['probability'][:, :4][waterData_1ps['Q1']['vcc']['kbarMask']][80000:160000],
                                     kMean_VCC=waterData_1ps['Q1']['vcc']['kbar'][waterData_1ps['Q1']['vcc']['kbarMask']][80000:160000],
                                     probability_Both=waterData_1ps['Q1']['both']['probability'][:, :4][waterData_1ps['Q1']['both']['kbarMask']][80000:160000],
                                     kMean_Both=waterData_1ps['Q1']['both']['kbar'][waterData_1ps['Q1']['both']['kbarMask']][80000:160000],
                                     nroi=687383.0,
                                     effectiveOverlap=mu_1ps,
                                     deltaEffectiveOverlap=delta_mu_1ps),
                     XPCSutil.getISF(probability_CC=waterData_1ps['Q2']['cc']['probability'][:, :4][waterData_1ps['Q2']['cc']['kbarMask']][80000:160000],
                                     kMean_CC=waterData_1ps['Q2']['cc']['kbar'][waterData_1ps['Q2']['cc']['kbarMask']][80000:160000],
                                     probability_VCC=waterData_1ps['Q2']['vcc']['probability'][:, :4][waterData_1ps['Q2']['vcc']['kbarMask']][80000:160000],
                                     kMean_VCC=waterData_1ps['Q2']['vcc']['kbar'][waterData_1ps['Q2']['vcc']['kbarMask']][80000:160000],
                                     probability_Both=waterData_1ps['Q2']['both']['probability'][:, :4][waterData_1ps['Q2']['both']['kbarMask']][80000:160000],
                                     kMean_Both=waterData_1ps['Q2']['both']['kbar'][waterData_1ps['Q2']['both']['kbarMask']][80000:160000],
                                     nroi=1072803.0,
                                     effectiveOverlap=mu_1ps,
                                     deltaEffectiveOverlap=delta_mu_1ps),
                     ]

print("================================================================")
print("The ISF 1ps is {:.2f} +- {:.4f}  for Q1 and {:.2f} +- {:.4f}  for Q2".format(waterContrast_1ps[0][0], waterContrast_1ps[0][1],
                                                                                    waterContrast_1ps[1][0], waterContrast_1ps[1][1], ))
print("================================================================")

waterContrast_7ps = [XPCSutil.getISF(probability_CC=waterData_7ps['Q1']['cc']['probability'][:, :4][waterData_7ps['Q1']['cc']['kbarMask']][80000:160000],
                                     kMean_CC=waterData_7ps['Q1']['cc']['kbar'][waterData_7ps['Q1']['cc']['kbarMask']][80000:160000],
                                     probability_VCC=waterData_7ps['Q1']['vcc']['probability'][:, :4][waterData_7ps['Q1']['vcc']['kbarMask']][80000:160000],
                                     kMean_VCC=waterData_7ps['Q1']['vcc']['kbar'][waterData_7ps['Q1']['vcc']['kbarMask']][80000:160000],
                                     probability_Both=waterData_7ps['Q1']['both']['probability'][:, :4][waterData_7ps['Q1']['both']['kbarMask']][80000:160000],
                                     kMean_Both=waterData_7ps['Q1']['both']['kbar'][waterData_7ps['Q1']['both']['kbarMask']][80000:160000],
                                     nroi=687383.0,
                                     effectiveOverlap=mu_7ps,
                                     deltaEffectiveOverlap=delta_mu_7ps),
                     XPCSutil.getISF(probability_CC=waterData_7ps['Q2']['cc']['probability'][:, :4][waterData_7ps['Q2']['cc']['kbarMask']][80000:160000],
                                     kMean_CC=waterData_7ps['Q2']['cc']['kbar'][waterData_7ps['Q2']['cc']['kbarMask']][80000:160000],
                                     probability_VCC=waterData_7ps['Q2']['vcc']['probability'][:, :4][waterData_7ps['Q2']['vcc']['kbarMask']][80000:160000],
                                     kMean_VCC=waterData_7ps['Q2']['vcc']['kbar'][waterData_7ps['Q2']['vcc']['kbarMask']][80000:160000],
                                     probability_Both=waterData_7ps['Q2']['both']['probability'][:, :4][waterData_7ps['Q2']['both']['kbarMask']][80000:160000],
                                     kMean_Both=waterData_7ps['Q2']['both']['kbar'][waterData_7ps['Q2']['both']['kbarMask']][80000:160000],
                                     nroi=1072803.0,
                                     effectiveOverlap=mu_7ps,
                                     deltaEffectiveOverlap=delta_mu_7ps),
                     ]

print("================================================================")
print("The ISF 7ps is {:.2f} +- {:.4f} for Q1 and {:.2f} +- {:.4f}  for Q2".format(waterContrast_7ps[0][0], waterContrast_7ps[0][1],
                                                                                   waterContrast_7ps[1][0], waterContrast_7ps[1][1], ))
print("================================================================")
