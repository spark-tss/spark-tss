package com.github.unsupervise.spark.tss.steps

import com.github.unsupervise.spark.tss
import com.github.unsupervise.spark.tss.core.TSS
import org.apache.spark.ml.linalg.SQLDataTypes
import org.apache.spark.sql.functions.{udf, element_at}

/**
  * Lazy Group By Scenarios, Order By Variable Name Step Command.
  * @param selectColNames the column to keep before grouping in source TSS, in addition to scenarioId, varName, row id, time definition and series columns
  * @param partitionsNumber the number of partitions for the output
  */
case class GroupScenariosStep(selectColNames: Array[String], partitionsNumber: Int) extends Step {

  override def apply(in: TSS): TSS = {
    import in.series.sparkSession.implicits._
    val udfVecToSeq = udf(tss.functions.vecToSeq)
    val selectCols = selectColNames.map(colName => {
      val colId = in.schema.fieldIndex(colName)
      val colType = in.schema.fields(colId).dataType
      if(colType == SQLDataTypes.VectorType){
        udfVecToSeq(in(colName)).alias(colName)
      }else{
        in(colName)
      }
    })
    val groupedMTSS =
      in.selectCols(Array(TSS.DECORATORS_COLNAME, TSS.ID_COLNAME, TSS.SERIES_COLNAME, TSS.TIMEFROM_COLNAME, TSS.TIMETO_COLNAME).map(in(_)) ++ selectCols)
        .group(in.getDecoratorColumn("simulationId"), in.getDecoratorColumn("varName"))
    groupedMTSS.selectCols((Seq(groupedMTSS(TSS.INGROUPKEY_COLNAME).alias("varName"), groupedMTSS(TSS.KEY_COLNAME).alias("scenarioId"), groupedMTSS(TSS.ID_COLNAME), groupedMTSS(TSS.SERIES_COLNAME), groupedMTSS(TSS.TIMEFROM_COLNAME), groupedMTSS(TSS.TIMETO_COLNAME), element_at(groupedMTSS("decorators"), 1).alias("decorators")) ++ selectColNames.map(groupedMTSS(_)))).coalesce(partitionsNumber).persist()
  }

}
