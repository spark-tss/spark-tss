# Spark Time Series Set (Spark-TSS)
Welcome to "Spark-TSS", a Big Data Univariate and Multivariate Time Series Clustering library written in Scala and based on Apache Spark.

## Include it in your SBT or Maven project

Add the following to your `build.sbt` file:

```scala
resolvers += Resolver.bintrayRepo("unsupervise", "maven")
libraryDependencies += "com.github.unsupervise" %% "spark-tss" % "0.2"
```

Or in your `pom.xml` file:

```xml
<dependencies>
<!-- Other Dependencies -->
    <dependency>
        <groupId>com.github.unsupervise</groupId>
        <artifactId>spark-tss_2.11</artifactId>
        <version>0.2</version>
    </dependency>
</dependencies>
<repositories>
<!-- Other Repositories ... -->
    <repository>
        <id>bintrayunsupervisemaven</id>
        <name>bintray-unsupervise-maven</name>
        <url>https://dl.bintray.com/unsupervise/maven/</url>
        <layout>default</layout>
    </repository>
</repositories>
```
## Execute your first Spark-TSS Snippet

Generate your first synthetic dataset with 2 time series families and build a basic clustering pipeline based on Fourier transforms.

```scala
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, element_at}

import com.github.unsupervise.spark.tss.benchmark.synthetic.DataGenerator
import com.github.unsupervise.spark.tss.benchmark.synthetic.FunPrototypes.{defaultSin, defaultSigmoid, defaultMorlet, defaultGaussian}
import com.github.unsupervise.spark.tss.core.TSS

object Main extends App {

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
    println("Accuracy: " + acc)

}
```
## Available Features

### Interpolation

Linear, Cubic

### Vectorization

- Discrete Fourier Transform (DFT)
- DFT Periodogram & Frequencies
- Inverse DFT
- Low Pass Filtering

### Distances & Related

- Euclidean Distances
- Get Closest Points to Target Points
- Barycenters Computation

### Dimension Reduction

- PCA, MDS

### (Co-)Clustering

- K-Means, GMM
- SOM (*with the help of the [SparkML-SOM package](https://github.com/FlorentF9/sparkml-som)*)
- Hierarchical Clustering

*+ with help of the [FunCLBM package](https://github.com/EtienneGof/FunCLBM):*
- LBM, CLBM, FunLBM, FunCLBM

### Evaluation 
*with help of the [Clustering4Ever](https://github.com/Clustering4Ever/Clustering4Ever) project:*

- Accuracy
- ARI
- Ball Hall
- Davies Bouldin
- F1
- Folkes-Mallows
- Jaccard
- Kulcztnski
- Mc Nemar
- MCC
- MI, NMI
- Purity
- Rand Index
- Recall
- Rogers Tanimoto
- Russel Rao
- Sokal Sneath

### Benchmarking

- Data Generation with embed 1D/2D (Co-)Clusters.
- Also Conditional Clustering Data Generation available.
- Possible prototypes for clusters: Sine, Sigmoids, Gaussians, Double Gaussians, Morlets, Rect, SinRect

### Misc. 

- Scaling, Centering, Z-Normalization, Sum to 1 Normalization
- One-Hot Encoding, Flattening, Grouping
- Contingency matrices
- Quantiles, Mean, StDev

## Authors

Anthony Coutant & Etienne Goffinet