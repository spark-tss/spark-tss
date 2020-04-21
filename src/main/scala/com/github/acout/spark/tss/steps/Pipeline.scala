package com.github.acout.spark.tss.steps

import com.github.acout.spark.tss.core.TSS

/**
  * A Pipeline is a sequence of Step objects allowing to differ their actual computation.
  * @param steps the Seq of Step objects to chain, in the adequate order.
  */
case class Pipeline(val steps: Seq[Step]) extends Step {
  /**
    * Actually run the computations for a given TSS object
    * @param in the input TSS to compute on
    * @return the output TSS, obtained after iteratively applying each step is the <steps> order.
    */
  def apply(in: TSS): TSS = steps.foldLeft(in){case (tss, step) => step(tss)}
  /**
    * Concatenate a Step at the end of the current Pipeline
    * @param nextStep the step to append
    * @return a Pipeline with <nextStep> added at the end of <this> Pipeline
    */
  override def +(nextStep: Step) = Pipeline(steps ++ Seq(nextStep))
}
