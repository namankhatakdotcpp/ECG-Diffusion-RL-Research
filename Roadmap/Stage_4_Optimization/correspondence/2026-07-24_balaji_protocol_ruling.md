From: Dr. Balaji
To: Naman
Subject: Re: Quick decisions on reliability-alpha sweep analysis & protocol

Naman,

Thanks for bringing this to my attention before committing anything or firing up more GPU jobs. Good call on holding off—pre-registration discipline means we don’t move goalposts or swap math definitions mid-flight.

Here are my direct answers to your 4 questions in Track A, plus my thoughts on your workflow in Track B.

My Responses to Track A Questions

1. A3 Mahalanobis Aggregation

Go with (b): distance computed from the mean reward / mean raw distance.

Option (a) taking log-inversions on per-sample rollouts before averaging is mathematically unstable. Non-zero epsilon cliffs blowing up a $+3.5\%$ shift into $+35.8\%$ is an artifact of the metric's form, not actual feature-space distortion. Base the $\le 8\%$ divergence check on the mean raw distance across the batch.

2. Condition 3 Scope & Metric Realignment

Use HYP-specific F1 as primary, with a strict Macro-F1 non-inferiority floor.

Evaluating a targeted diagnostic reward against non-target classes without separating them creates noise.

* Primary Target: We require an absolute improvement of $\ge +3.0\%$ on HYP-F1 to consider an arm effective for the target class.
* Secondary Safety Bound: Overall Macro-F1 must not drop by more than $1.5\%$ absolute. If it drops beyond $1.5\%$, the arm is disqualified due to collateral degradation across non-target classes.

3. Seed Protocol

Yes, literal Seeds 43 and 44. Run Seed 42 (our exploratory sweep), then verify the baseline vs. winning $\alpha^*$ on Seeds 43 and 44. Do not rely on relative list indices in `config.yaml`—hardcode the actual seed integers so the run logs are fully reproducible for reviewer verification.

4. "Stays Flat" Epsilon Band

Set $\epsilon = \pm 0.5\%$ absolute Macro-F1.

Anything within an absolute change of $\le 0.5\%$ is expected evaluation noise and should be formally classified as "statistically flat / unchanged."

Regarding the Fallback to $\alpha = 1.00$:

I am 100% comfortable with $\alpha = 1.00$ remaining the default.

If none of the relaxation arms ($\alpha \in \{0.75, 0.50, 0.25, 0.00\}$) beat the baseline while satisfying all non-inferiority constraints, that is still a valid scientific finding. It proves the original reliability weighting was necessary and optimal. Never force a hyperparameter change just to claim a "new proposed method."
