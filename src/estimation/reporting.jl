# For a full report we need
# - A nicely formated model formulated
# - Its initial configuration
# ---
# - The method used to approximate the likelihood
# - Show nth trace of param, grad, f_eval
# - Number of observations
# - Nlog(2pi)
# - objective
# - objective + constant
# - time (?)

function report_model(fmt)
  # show everything model-y
end
function report_model_param(fmt)
  # show initial values
  # show domains
end
function report_data(fmt)
  # how much data
  # how many subjects
end
function report_trace(fmt::FittedPuMaSModel; show_every=5)
    tr = Optim.trace(fmt.optim)
    for i = 1:show_every:length(tr)
        data = tr[i].metadata
        println("iteration: ", i-1)
        println("Objectiv value: ", tr[i].value)
        println("x:    ", data["x"])
        println("g(x): ", data["g(x)"])
        println()
    end
    if length(tr) % show_every == 0
        data = tr[end].metadata
        println("iteration: ", length(tr)-1)
        println("Objectiv value: ", tr[end].value)
        println("x:    ", data["x"])
        println("g(x): ", data["g(x)"])
        println()
    end
end
function report_fit(fmt)
  # show estimates
  # show etabar
  # show various likelihood info
end
function report_cov(fmt)
  # show different varcov matrices
end
