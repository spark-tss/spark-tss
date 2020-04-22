# spark-tss
Spark Time Series Set data analysis

## Getting Started

Add the following to your `build.sbt` file:

```scala
resolvers += Resolver.bintrayRepo("anthonycoutant", "SparkTSS")
libraryDependencies += "com.github.unsupervise" %% "spark-tss" % "0.1"
```

Or in your `pom.xml` file:

```xml
<dependencies>
<!-- Other Dependencies -->
    <dependency>
        <groupId>com.github.unsupervise</groupId>
        <artifactId>spark-tss_2.11</artifactId>
        <version>0.1</version>
        <type>pom</type>
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