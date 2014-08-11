(ns ch6-naive-bayes.core-test
  (:require [clojure.test :refer :all]
            [ch6-naive-bayes.core :refer :all]))

(deftest feature-test
  (testing "feature increment and probability"
    (let [counter (increment-feature {:rabbit {:good 2 :bad 3}} "rabbit" :good)
          prob    (feature-probability {:rabbit {:good 2 :bad 3}} "rabbit" :good)]
    (is (= counter {:rabbit {:good 3 :bad 3}}))
    (is (= prob (/ 2 5))))))
