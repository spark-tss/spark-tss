package com.github.unsupervise.spark.tss.steps

import java.io.{BufferedWriter, FileWriter}

import com.github.unsupervise.spark.tss.core.TSS
import com.github.unsupervise.spark.tss.functions
import org.apache.spark.ml.linalg.SQLDataTypes
import org.apache.spark.sql

/**
  * Lazy Clustering Evaluation Step Command computing several output from a clustering column in a TSS object.
  * The class is designed to perform several clustering computations at once, thus has only Array parameters.
  * @param inClusteringColNames the Array of clustering column name from which computing evaluations
  * @param inRepresentationColNames the Array of representation column names from which the respective clusterings in <inClusteringColNames> have been performed
  * @param centersInFilePaths the Array of clustering center file paths, for the respective clustering column names in <inClusteringColNames>
  * @param contingencyMatrixOutFilePaths the Array of output decorators to clustering contingency matrices file paths for the respective clustering column names in <inClusteringColNames>
  * @param daviesBouldinOutFilePath the Array of output Davies-Bouldin indices file paths for the respective clustering column names in <inClusteringColNames>
  */
case class ClusteringEvaluationStep(inClusteringColNames: Array[String], inRepresentationColNames: Array[String], centersInFilePaths: Array[String], contingencyMatrixOutFilePaths: Array[String], daviesBouldinOutFilePath: Array[String]) extends Step {

  override def apply(in: TSS): TSS = {

    implicit val ss = in.series.sparkSession
    import in.series.sparkSession.implicits._

    //Examine Multivariate Clustering Results for each performed one:
    inClusteringColNames.indices.foreach(i => {
      val label = inClusteringColNames(i)
      in.contingencyMatrixMapScalar(
        TSS.DECORATORS_COLNAME + "_key", TSS.DECORATORS_COLNAME + "_value", "cluster", "count",
        TSS.DECORATORS_COLNAME, label
      ).write.mode("overwrite").parquet(contingencyMatrixOutFilePaths(i))

      //Compute Clustering Evaluation
      val evalIndex = if(in.series.schema(inRepresentationColNames(i)).dataType == SQLDataTypes.VectorType){
        in.addUDFColumn(inRepresentationColNames(i) + "Seq", inRepresentationColNames(i), sql.functions.udf(functions.vecToSeq)).daviesBouldin(inRepresentationColNames(i) + "Seq", label)
      }else{
        in.daviesBouldin(inRepresentationColNames(i), label)
      }
      val evalWriter = new BufferedWriter(new FileWriter(daviesBouldinOutFilePath(i)))
      evalWriter.write(evalIndex.toString)
      evalWriter.close()
    })

    //Return input TSS, for chaining capacities with other steps
    in
  }

}
