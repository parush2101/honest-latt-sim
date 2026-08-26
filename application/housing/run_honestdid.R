## Run the actual HonestDiD package on the pooled shale/house-price event study,
## to check the ATT sensitivity interval reported in the paper's Section 6.
suppressMessages(library(HonestDiD)); library(jsonlite)

betahat <- as.numeric(read.csv("housing/betahat.csv", header = FALSE)[[1]])
sigma   <- as.matrix(read.csv("housing/sigma.csv",   header = FALSE))
dimnames(sigma) <- NULL
meta    <- fromJSON("housing/meta.json")
numPre  <- meta$numPre; numPost <- meta$numPost; l_vec <- matrix(meta$l_vec, ncol = 1)

cat(sprintf("betahat = %s\n", paste(round(betahat,4), collapse=", ")))
cat(sprintf("numPre=%d numPost=%d  l_vec=%s\n", numPre, numPost, paste(l_vec, collapse=",")))
att <- as.numeric(t(l_vec) %*% betahat[(numPre+1):(numPre+numPost)])
seA <- sqrt(as.numeric(t(l_vec) %*% sigma[(numPre+1):(numPre+numPost),(numPre+1):(numPre+numPost)] %*% l_vec))
cat(sprintf("\nATT (l'beta_post) = %+.4f   se = %.4f   naive 95%% CI = [%+.4f, %+.4f]\n\n",
            att, seA, att-1.96*seA, att+1.96*seA))

## ---- original CS (no restriction; sampling only) ----
orig <- constructOriginalCS(betahat, sigma, numPre, numPost, l_vec = l_vec)
cat(sprintf("Original CS (sampling only): [%+.4f, %+.4f]\n\n", orig$lb, orig$ub))

## ---- Delta^RM (relative magnitudes): post violation <= Mbar * max pre violation ----
cat("Delta^RM(Mbar): post-treatment violation bounded by Mbar x max pre-treatment violation\n")
rm <- createSensitivityResults_relativeMagnitudes(
        betahat, sigma, numPre, numPost, l_vec = l_vec,
        Mbarvec = seq(0, 2, 0.1))
show <- rm[rm$Mbar %in% c(0, 0.5, 1, 1.5, 2), c("Mbar","lb","ub")]
print(show, row.names = FALSE)
inc0 <- rm$lb <= 0 & rm$ub >= 0
Mbar_star <- if (any(inc0)) rm$Mbar[which(inc0)[1]] else NA
cat(sprintf("  -> ATT honest CI EXCLUDES 0 for Mbar < %.1f; first admits 0 at Mbar* = %.1f\n\n",
            Mbar_star, Mbar_star))

## ---- Delta^SD (smoothness): guarded (CVXR>=1.0 breaks findOptimalFLCI) ----
sd <- tryCatch(
  createSensitivityResults(betahat, sigma, numPre, numPost, l_vec = l_vec,
                           Mvec = c(0, 0.01, 0.02, 0.03)),
  error = function(e) NULL)
if (!is.null(sd)) { cat("Delta^SD(M):\n"); print(sd[, c("M","lb","ub")], row.names = FALSE) } else {
  cat("Delta^SD skipped (installed CVXR incompatible with HonestDiD's findOptimalFLCI); ",
      "Delta^RM above is the relative-magnitudes result the paper cites.\n") }
