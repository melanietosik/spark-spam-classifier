from pyspark import SparkConf, SparkContext

from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.classification import LogisticRegressionWithSGD

conf = SparkConf().setAppName("spam-classifier")
sc = SparkContext(conf=conf)

log4j = sc._jvm.org.apache.log4j
log4j.LogManager.getRootLogger().setLevel(log4j.Level.ERROR)

spam = sc.textFile("spam.txt")
ham = sc.textFile("ham.txt")

# Create a HashingTF instance to map email text to vectors of 10,000 features.
tf = HashingTF(numFeatures=10000)

# Each email is split into words, and each word is mapped to one feature.
spamFeatures = spam.map(lambda email: tf.transform(email.split(" ")))
hamFeatures = ham.map(lambda email: tf.transform(email.split(" ")))

# Create LabeledPoint datasets for positive (spam) and negative (ham) examples.
positiveExamples = spamFeatures.map(lambda features: LabeledPoint(1, features))
negativeExamples = hamFeatures.map(lambda features: LabeledPoint(0, features))

trainingData = positiveExamples.union(negativeExamples)
trainingData.cache()  # Cache since Logistic Regression is an iterative algorithm.

# Run Logistic Regression using the SGD algorithm.
model = LogisticRegressionWithSGD.train(trainingData)

# Test examples
test = [
    "O M G GET cheap stuff by sending money",
    "Hi Dad, how is the family?",
    "hey what's up, I'm almos there",
    "Reply 1000 to win a prize of a million dollars",
    "dude get down here it's urgent",
    "Wanna go see Star Wars tonight",
]

# Predict
for x in test:
    vec = tf.transform(x.split(" "))
    print(model.predict(vec), x)
