package com.github.unsupervise.spark.tss.steps

import com.github.unsupervise.spark.tss.core.TSS

/**
  * Lazy Persist Step Command.
  */
object PersistStep extends Step {

  /**
    * Actually apply the Step to a given TSS
    *
    * @param in the input TSS
    * @return a TSS being the result of the Step computation
    */
  override def apply(in: TSS): TSS = {
    in.persist()
  }

}
