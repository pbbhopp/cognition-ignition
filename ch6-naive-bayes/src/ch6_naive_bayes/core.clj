(ns ch6-naive-bayes.core)

(defprotocol Filter
  (increment-feature [data feature category])
  (increment-category [data category])
  (category-count [data category])
  (feature-count [data feature category]))

(extend-type clojure.lang.PersistentArrayMap
  Filter
  (increment-feature [data feature category]
    (let [result  ((keyword feature) data)
          updated (merge-with + {:count 1 :c 1} result)]
      (assoc data :word updated))))

(defprotocol Feature
  (get-words [str]))

(extend-type String
  Feature
  (get-words [str] (apply hash-set (clojure.string/split str #" "))))