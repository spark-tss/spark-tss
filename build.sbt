//import SparkSubmit.settings

name := "spark-tss"

version := "0.1"

scalaVersion := "2.11.12"

//crossScalaVersions := Seq("2.11.12","2.10.6")

publishMavenStyle := true

libraryDependencies += "org.apache.spark" %% "spark-core" % "2.4.3"
libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.4.3"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.4.3"

libraryDependencies += "com.github.haifengl" %% "smile-scala" % "1.5.1"
libraryDependencies += "com.github.haifengl" % "smile-netlib" % "1.5.1"

libraryDependencies  ++= Seq(
  // Last stable release
  "org.scalanlp" %% "breeze" % "0.13.2",

  // Native libraries are not included by default. add this if you want them (as of 0.7)
  // Native libraries greatly improve performance, but increase jar sizes.
  // It also packages various blas implementations, which have licenses that may or may notBon
  // be compatible with the Apache License. No GPL code, as best I know.
  "org.scalanlp" %% "breeze-natives" % "0.13.2",

  // The visualization library is distributed separately as well.
  // It depends on LGPL code
  "org.scalanlp" %% "breeze-viz" % "0.13.2"
)

//BLAS / LAPACK
libraryDependencies += "com.github.fommil.netlib" % "all" % "1.1.2"

//TESTS
libraryDependencies += "org.scalactic" %% "scalactic" % "3.0.5"
libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.5" % "test"

resolvers += "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"
//resolvers += "Artima Maven Repository" at "http://repo.artima.com/releases"

//CLUSTERING
libraryDependencies += "org.clustering4ever" % "clustering4ever_2.11" % "0.9.6"
resolvers += Resolver.bintrayRepo("clustering4ever", "C4E")

resolvers += Resolver.sonatypeRepo("snapshots")

libraryDependencies += "xyz.florentforest" %% "sparkml-som" % "0.1"

//libraryDependencies += "com.LIPN" %% "bopt" % "1.0"
//libraryDependencies += "com.github.yazidjanati" %% "bopt" % "0.2.1"

assemblyMergeStrategy in assembly := {
  case PathList("META-INF", xs @ _*) => MergeStrategy.discard
  case x => MergeStrategy.first
}

test in assembly := {}

organization := "com.github.acout"
licenses += ("Apache-2.0" -> url("http://www.apache.org/licenses/LICENSE-2.0.txt"))
bintrayRepository := "SparkTSS"
bintrayOrganization := Some("anthonycoutant")