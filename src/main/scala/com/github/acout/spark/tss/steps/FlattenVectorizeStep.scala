package com.github.acout.spark.tss.steps
import com.github.acout.spark.tss.core.TSS

/**
  * Lazy Flattening and Vectorization Step Command.
  * The class is designed to perform several computations at once, thus works with Array parameters.
  * @param outFlattenedSeqColNames the Array of output flattened (but not vectorized, just Seq) column names
  * @param outFlattenedVectorColNames the Array of output flattened and vectorized (thus of Spark ML Vector type) column names
  * @param inColNames the Array of input column names to which iteratively applying the flat and vectorize operations
  */
case class FlattenVectorizeStep(outFlattenedSeqColNames: Array[String], outFlattenedVectorColNames: Array[String], inColNames: Array[String]) extends Step {
  /**
    * Actually apply the Step to a given TSS
    *
    * @param in the input TSS
    * @return a TSS being the result of the Step computation
    */
  override def apply(in: TSS): TSS = {
    outFlattenedSeqColNames.indices.foldLeft(in) {
      case (tss, i) => {
        tss.addFlatten(outFlattenedSeqColNames(i), inColNames(i)).addMLVectorized(outFlattenedVectorColNames(i), outFlattenedSeqColNames(i))
      }
    }
  }
}
