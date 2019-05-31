"""
      percentage_prediction_error(t, C_act,C_pre)
      where, C_act is original plasma concentration profile
             C_pre is predicted plasma concentration profile (using IVIVC Correlation model)

      It returns percentage prediction error of Cmax and AUClast (till the last time point)
      PE of Cmax and AUClast should not exceed 15% for each formulation and average should not
      exceed 10%
"""

# Internal Validaton
function percentage_prediction_error(t, C_act, C_pre)
  sub_act = NCASubject(C_act, t)
  sub_pre = NCASubject(C_pre, t)
  # AUC Prediction Error
  auc_act = NCA.auc(sub_act, auctype=:last)
  auc_pre = NCA.auc(sub_pre, auctype=:last)
  auc_pe = (auc_act - auc_pre)/auc_act * 100.0
  # Cmax Prediction Error
  cmax_act = NCA.cmax(sub_act)
  cmax_pre = NCA.cmax(sub_pre)
  cmax_pe = (cmax_act - cmax_pre)/cmax_act * 100.0
  # return cmax_pe and auc_pe
  cmax_pe, auc_pe
end

# External Validaton