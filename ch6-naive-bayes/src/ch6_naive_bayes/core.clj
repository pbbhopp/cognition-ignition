(ns ch6-naive-bayes.core)

(defprotocol Filter
  (increment-feature [data feature category])
  (increment-category [data category])
  (category-count [data category])
  (feature-count [data feature category])
  (feature-probability [data feature category]))

(extend-type clojure.lang.PersistentArrayMap
  Filter
  (increment-feature [data feature category]
    (let [result  ((keyword feature) data)
          updated (merge-with + {(keyword category) 1} result)]
      (assoc data (keyword feature) updated)))
  (feature-probability [data feature category]
    (let [result ((keyword feature) data)
          sum    (reduce + (vals result))
          count  (category result)]             
      (if (= count 0)
        0
        (/ count sum)))))

(defprotocol Feature
  (get-words [str]))

(extend-type String
  Feature
  (get-words [str] (apply hash-set (clojure.string/split str #" "))))