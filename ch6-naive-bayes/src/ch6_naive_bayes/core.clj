(ns ch6-naive-bayes.core)

(defprotocol Filter
  (increment-feature [data-set feature category])
  (increment-category [data-set category])
  (category-count [data-set category])
  (feature-count [data-set feature category]))

(extend-type clojure.lang.PersistentHashSet
  Filter
  (increment-feature [data-set feature category]
    (let [result  (clojure.set/select #(= (:word %) feature) data-set)
          updated (merge-with + {:count 1 :c 1} (:counts (first result)))]
      updated)))

(defprotocol Feature
  (get-words [str]))

(extend-type String
  Feature
  (get-words [str] (apply hash-set (clojure.string/split str #" "))))