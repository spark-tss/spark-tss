# spark-tss
Spark Time Series Set data analysis

## Getting Started

Add the following to your `build.sbt` file:

```scala
resolvers += Resolver.bintrayRepo("anthonycoutant", "SparkTSS")
libraryDependencies += "com.github.acout" %% "spark-tss" % "0.1"
```

Or in your `pom.xml` file:

```xml
<dependencies>
<!-- Other Dependencies -->
    <dependency>
        <groupId>com.github.acout</groupId>
        <artifactId>spark-tss_2.11</artifactId>
        <version>0.1</version>
    </dependency>
</dependencies>
<repositories>
<!-- Other Repositories ... -->
    <repository>
        <id>bintrayanthonycoutantSparkTSS</id>
        <name>bintray-anthonycoutant-SparkTSS</name>
        <url>https://dl.bintray.com/anthonycoutant/SparkTSS/</url>
        <layout>default</layout>
    </repository>
</repositories>
```