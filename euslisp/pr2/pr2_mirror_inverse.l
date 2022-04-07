#!/usr/bin/env roseus

(load "package://imitation_learning_pr2/euslisp/pr2_utils.l")

(unless (boundp '*ri*)
  (progn
    (require :pr2-interface "package://pr2eus/pr2-interface.l")
    (pr2-init)
    (setq *co* (make-coords))
    (objects (list *pr2*))
    )
  )

(defun initial-pose (&key (r-sholder-pan-ja -122))
  (send *pr2* :reset-manip-pose)
  (send *pr2* :r_shoulder_pan_joint :joint-angle r-sholder-pan-ja)
  (send *ri* :angle-vector (send *pr2* :angle-vector) 4000)
  (send *ri* :wait-interpolation)
  (send *viewer* :draw-objects)

  (send *ri* :stop-grasp)
  )

(defun initial-pose-tmp (&key (offset 90))
  (send *pr2* :reset-manip-pose)
  ;; (send *co* :newcoords (make-coords :pos #f(550 150 1000) :rpy #f(0 0 -1.571))) :: hook bag
  (send *co* :newcoords (make-coords :pos #f(550 150 1000) :rpy #f(-1.571 0 -1.571))) ;; cap
  (send *pr2* :larm :inverse-kinematics
        *co*
        :rotation-axis t
        :revert-if-fail nil)
  (setq lav (send *pr2* :larm :angle-vector))
  (setq dif-for-invert-vec (scale 2 (float-vector (elt lav 0) 0 (elt lav 2) 0 (elt lav 4) 0 (elt lav 6))))
  (setq n-lav (v- lav dif-for-invert-vec))
  (setq dif-for-adj-vec (float-vector offset 0 0 0 0 0 0))
  (setq rav (v- n-lav dif-for-adj-vec))

  (send *pr2* :rarm :angle-vector rav)
  (send *pr2* :head-neck-y :joint-angle 0.0)
  (send *pr2* :head-neck-p :joint-angle 60.0)
  (send *ri* :angle-vector (send *pr2* :angle-vector) 4000)
  (send *ri* :wait-interpolation)
  (send *viewer* :draw-objects)

  ;; (send *ri* :stop-grasp) ;;hook bag
  (send *ri* :start-grasp) ;;hook cap
  )

(defun manequine-set ()
  (manequine-mode :arm (list "larm") :start nil) ;;rarmはマネキンoff (onでも良い？)
  (manequine-mode :arm (list "rarm") :start t) ;;larmはマネキンon
  )

(defun manequine-end ()
  (manequine-mode :arm (list "rarm" "larm") :start nil) ;;マネキンoff
  )

(defun mirror (&key (offset 45))
  ;; get now angle-vector
  (send *pr2* :angle-vector (send *ri* :state :potentio-vector))
  (send *pr2* :head-neck-y :joint-angle 0.0)
  (send *pr2* :head-neck-p :joint-angle 60.0)
  (setq lav (send *pr2* :rarm :angle-vector))

  ;; convert to larm angle-vector (inverts data with index 0,2,4)
  (setq dif-for-adj-vec (float-vector offset 0 0 0 0 0 0))
  (setq n-lav (v+ lav dif-for-adj-vec))
  ;; (setq dif-for-invert-vec (scale 2 (float-vector (elt n-lav 0) 0 (elt n-lav 2) 0 (elt n-lav 4) 0 0)))
  (setq dif-for-invert-vec (scale 2 (float-vector (elt n-lav 0) 0 (elt n-lav 2) 0 (elt n-lav 4) 0 (elt n-lav 6))))
  (setq rav (v- n-lav dif-for-invert-vec))
  (format t "lav : ~A , rav : ~A ~%" lav rav)
  ;; (setq inverts-data-list (list 0 2 4))
  ;; (dotimes (i (length lav))
  ;;   (if (= i 0)
  ;; 	 (setq (elt av i) (- (elt lav i) 45))
  ;;     )
  ;;   (if (member i inverts-data-list)
  ;; 	 (setq (elt av i) (* -1 (elt av i)))
  ;;     ))

  ;; send angle-vector
  (send *pr2* :larm :angle-vector rav)
  ;; (send *ri* :angle-vector (send *pr2* :angle-vector) 1000)
  (send *ri* :angle-vector-sequence (list (send *pr2* :angle-vector)) (list 400))
  ;; (send *ri* :wait-interpolation nil 0.01)
  (send *ri* :move-gripper :larm (* (send *ri* :state :gripper :rarm :position) 0.001) :wait t)
  )

(defun mirror-tmp (&key (offset 90))
  ;; larm の angle-vectorをrarmに移す
  (manequine-end)
  (initial-pose-tmp)
  (enable-gripper-manequine-mode)

  ;; マネキンモードのセット
  (manequine-set)

  ;; angle-vector mirrorのループ
  (start-data-collection)
  (unix:sleep 3)
  (do-until-key
   (mirror :offset offset)
   )

  (end-data-collection)
  (manequine-end)
  )

(defun mirror-tmp-dialogue (&key (offset 90))
  ;; larm の angle-vectorをrarmに移す
  (manequine-end)
  (initial-pose-tmp)
  (enable-gripper-manequine-mode)

  (send *ri* :speak-jp "準備をして合図をして下さい")
  (ok-wait)
  (initial-pose-tmp)
  (enable-gripper-manequine-mode)

  ;; マネキンモードのセット
  (manequine-set)

  (send *ri* :speak-jp "データを記録します")
  ;; angle-vector mirrorのループ
  (start-data-collection)
  ;; (start-data-collection :service "long_data_collection_rosbag_test/start_request")
  (unix:sleep 3)
  ;; (ros::unsubscribe "/speech_to_text")
  ;; (setq *ok-flag* nil)
  ;; (ros::subscribe "/speech_to_text" speech_recognition_msgs::SpeechRecognitionCandidates #'voice-ok-cb)
  (ros::rate 20);;10
  ;; (until *ok-flag*
  (do-until-key
         (mirror :offset offset)
         ;; (ros::spin-once)
         ;; (ros::sleep)
         )

  (end-data-collection)
  ;; (send *ri* :stop-grasp :larm)
  ;; (do-until-key
  ;;        (mirror :offset offset)
  ;;        ;; (ros::spin-once)
  ;;        ;; (ros::sleep)
  ;;        )
  ;; (start-data-collection :service "long_data_collection_rosbag_test/end_request")

  (send *ri* :speak-jp "データを保存します")
  (end-data-collection)
  (manequine-end)
  )

(defun mirror-tmp-dialogue-loop ()
  (let ((end-flag nil))
    (send *ri* :speak-jp "データ収集を開始します")

    (until end-flag
           (mirror-tmp-dialogue)
           ;; (setq end-flag (ask-end))
           )
    ))


;; record and replay test
(defun record-lav-key ()
  (manequine-mode :arm (list "larm") :start t)
  (setq *ok-flag* nil)
  (setq  lav-list (list))
  (do-until-key
   (send *pr2* :angle-vector (send *ri* :state :potentio-vector))
   (setq now-lav (send *pr2* :larm :angle-vector))
   (format t "now lav : ~A" now-lav)
   (setq lav-list (append lav-list (list now-lav)))
   (unix:usleep 100000)
   )
  (manequine-end)
  (setq lav-length (length lav-list))
  (format t "length of lav-list : ~A ~%" lav-length)
  lav-list
  )

(defun replay-lav-list (lav-list &key (time 200) (reverse nil))
  (if reverse
      (setq lav-list (reverse lav-list))
    )
  (setq lav-length (length lav-list))
  (format t "length of lav-list : ~A ~%" lav-length)
  (send *pr2* :angle-vector (send *ri* :state :potentio-vector))
  (setq now-head-y (send *pr2* :head-neck-y :joint-angle))
  (setq now-head-p (send *pr2* :head-neck-p :joint-angle))

  (setq avs (list))
  (setq tms (list))
  (dolist (av lav-list)
    ;; (send *pr2* :head-neck-y :joint-angle now-head-y)
    ;; (send *pr2* :head-neck-p :joint-angle now-head-p)
    ;; (send *pr2* :head-neck-y :joint-angle -0.005299)
    ;; (send *pr2* :head-neck-p :joint-angle 52.746)
    (send *pr2* :head-neck-y :joint-angle 0.0)
    (send *pr2* :head-neck-p :joint-angle 60.0)
    (send *pr2* :larm :angle-vector av)
    (setq avs (append avs (list (send *pr2* :angle-vector))))
    (setq tms (append tms (list time)))
    )
  (send *ri* :angle-vector-sequence avs tms)
  (send *ri* :wait-interpolation)
  )

(defun tmp-test ()
  (manequine-mode :arm (list "larm") :start nil)
  (initial-pose-tmp)
  (setq tmp-list (record-lav-key))
  (initial-pose-tmp)
  (replay-lav-list tmp-list)
  )
