package com.github.unsupervise.spark.tss.core

import org.scalatest.FlatSpec

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, element_at}

import com.github.unsupervise.spark.tss.benchmark.synthetic.DataGenerator
import com.github.unsupervise.spark.tss.benchmark.synthetic.FunPrototypes.{defaultSin, defaultSigmoid, defaultMorlet, defaultGaussian}
import com.github.unsupervise.spark.tss.core.TSS

class ReadMeSpec extends FlatSpec {

    "README file" should "have working getting started snippet" in {
        //Create SparkSession which will be implictly given to functions
        implicit val ss = SparkSession
        .builder()
        .master("local[*]")
        .appName("Spark-TSS Test")
        .config("spark.executor.cores", 2)
        .config("spark.executor.heartbeatInterval", "20s")
        .config("spark.driver.memory", "2G")
        .getOrCreate()
        //Assign log level and checkpoint dirs
        ss.sparkContext.setCheckpointDir("checkpointDir")
        ss.sparkContext.setLogLevel("WARN")
        //Define data generation prototypes (2 observation and 2 feature clusters)
        val prototypes = List(List(defaultSin, defaultMorlet), List(defaultSigmoid, defaultGaussian))
        //Generate data for an observation x features matrix of 400 x 10 with half observations and columns in each cluster
        val syntheticTSS = DataGenerator.random2DClusters(prototypes, 0.01, List(200, 200), List(5, 5))
        //Observe TSS Dataframe columns. "series" contains the univariate series corresponding to previously generated multivariate series.
        //Univariate series of a same Multivariate series are differenciated with the decorators.varName id
        syntheticTSS.orderBy("id").show(200)
        //Compute Fourier log10 periodograms over Z-normalized version of them
        val fourierPeriodogramsTSS = 
            syntheticTSS.addZNormalized("zseries", TSS.SERIES_COLNAME, 0.0001)
                        .addDFT("dft", "zseries")
                        .addDFTPeriodogram("rawPeriodogram", "dft")
                        .addLog10("logPeriodogram", "rawPeriodogram", 1D)
                        .addMLVectorized("logPeriodogramVec", "logPeriodogram")
                        .addColScaled("preProcessedPeriodogramVec", "logPeriodogramVec", true, true)
        //Observe how the "addXXX" methods create new columns in the TSS underlying DataFrame, which names are the first parameters of the associated calls
        fourierPeriodogramsTSS.show(7)
        //Perform PCA Dimension Reduction on Fourier Periodograms, with 5 components
        val reducedTSS = fourierPeriodogramsTSS.addPCA("pcaVec", "preProcessedPeriodogramVec", 2).addSeqFromMLVector("pca", "pcaVec")
        reducedTSS.select("pca", "trueSeriesCluster").repartition(1).saveCSV("./pca_out.csv", true, true)
        //Group series by multivariate series
        val multivariateTSS = reducedTSS.group(reducedTSS.getDecoratorColumn(TSS.SIMULATIONID_DECORATORNAME), reducedTSS.getDecoratorColumn(TSS.VARNAME_DECORATORNAME)).withColumn("trueCluster", element_at(col("trueSeriesCluster"), 1))
        multivariateTSS.select("gkey", "inGKey", "pca", "trueCluster").orderBy("gkey").show(200)//repartition(1).saveCSV("./multi_out.csv", true, true)
        //Call LBM co-clustering method                 
        val clusteredMTSS = multivariateTSS.addLBM("cluster", "pca", "gkey", TSS.SERIES_COLNAME, 2, 2, 10, 5, "randomPartition", "SEMGibbs", true, 1, 3)
        //Observe clustering results against true ones stored by data generation step
        clusteredMTSS.select("cluster", "trueCluster").show(7)
        //Evaluate clustering results using Accuracy
        val acc = clusteredMTSS.accuracy("cluster", "trueCluster")
        assert((acc - 1.0) < 0.001)
    }

    
}