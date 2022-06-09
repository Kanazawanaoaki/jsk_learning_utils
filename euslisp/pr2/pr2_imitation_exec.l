#!/usr/bin/env roseus

(load "package://jsk_learning_utils/euslisp/pr2/pr2_utils.l")

(unless (boundp '*ri*)
  (progn
    (require :pr2-interface "package://pr2eus/pr2-interface.l")
    (pr2-init)
    (setq *co* (make-coords))
    (objects (list *pr2*))
    )
  )

;; for ros
;; angle-vector topic load-ros-manifest

;; subscriber


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

  (send *ri* :stop-grasp)
  )

(defun initial-pose-rcup ()
  (setq *initial-av* #f(299.873 75.0006 49.9994 110.003 -110.003 -20.0005 -10.0016 -10.0014 -24.3784 46.5655 -44.1727 -118.165 168.29 -67.9053 -241.519 0.0 60.0))
  (send *pr2* :angle-vector *initial-av*)
  (send *ri* :angle-vector (send *pr2* :angle-vector))
  (send *ri* :wait-interpolation)
  (send *ri* :stop-grasp)
  )

(defun initial-pose-blue-box ()
  (setq *initial-av* '#f(299.831 74.9721 49.9994 109.993 -110.003 -20.0038 -10.0054 -9.99972 -33.8218 2.75405 -59.0934 -106.486 270.317 -114.592 -201.405 0.000701 61.356))
  (send *pr2* :angle-vector *initial-av*)
  (send *ri* :angle-vector (send *pr2* :angle-vector))
  (send *ri* :wait-interpolation)
  (send *ri* :stop-grasp)
  )

;; initial-pose-tmp -> move with imitation
(defun imitation-cmd-cb (msg)
  (setq *msg* msg)
  (setq *now-cmd* (send *msg* :data))
  (format t "now angle-vector ~A~%" *now-cmd*)
  )


(defun start-imitation ()
  ;; (send *ri* :speak-jp "学習した動作を実行します")
  (ros::subscribe "/imitation/command" std_msgs::Float64MultiArray #'imitation-cmd-cb)
  (setq *now-cmd* (send *pr2* :rarm :angle-vector))
  (ros::rate 5);;10
  (do-until-key
   ;; subscribeして動作する．
   (send *pr2* :head-neck-y :joint-angle 0.0)
   (send *pr2* :head-neck-p :joint-angle 60.0)
   (setq now-rarm-cmd (map float-vector #'(lambda (x) (rad2deg x)) *now-cmd*))
   (format t "now rarm cmd ~A ~%" now-rarm-cmd)
   (send *pr2* :rarm :angle-vector now-rarm-cmd)
   ;; (send *pr2* :r_forearm_roll_joint :joint-angle (+ (send *pr2* :r_forearm_roll_joint :joint-angle) 90))
   ;; (send *pr2* :r_wrist_roll_joint :joint-angle (+ (send *pr2* :r_wrist_roll_joint :joint-angle) 90))

   ;; (send *ri* :angle-vector (send *pr2* :angle-vector))
   (send *ri* :angle-vector-sequence (list (send *pr2* :angle-vector)) (list 400)) ;;400

   ;; (send *ri* :wait-interpolation nil 0.01)
   (send *viewer* :draw-objects)
   (ros::spin-once)
   (ros::sleep)
   )
  (send *ri* :start-grasp :rarm)
  )

(defun start-imitation-cap ()
  ;; (send *ri* :speak-jp "学習した動作を実行します")
  (ros::unsubscribe "/imitation/command")
  (ros::subscribe "/imitation/command" std_msgs::Float64MultiArray #'imitation-cmd-cb)
  ;; (setq *now-cmd* (send *pr2* :rarm :angle-vector))
  (setq *now-cmd* nil)
  (ros::rate 5);;10
  (do-until-key
   (when *now-cmd*
     ;; subscribeして動作する．
     (send *pr2* :head-neck-y :joint-angle 0.0)
     (send *pr2* :head-neck-p :joint-angle 60.0)
     ;; (setq now-larm-cmd (subseq *now-cmd* 0 7))
     (setq now-larm-cmd (map float-vector #'(lambda (x) (rad2deg x)) (subseq *now-cmd* 0 7)))
     (setq now-l-gripper-cmd (elt *now-cmd* 7))
     (format t "now larm cmd ~A ~%" now-larm-cmd)
     (send *pr2* :larm :angle-vector now-larm-cmd)
     ;; (send *ri* :angle-vector (send *pr2* :angle-vector))
     (send *ri* :angle-vector-sequence (list (send *pr2* :angle-vector)) (list 400)) ;;200
     (format t "now l-gripper cmd ~A ~%" now-l-gripper-cmd)
     (send *ri* :move-gripper :larm  now-l-gripper-cmd)

     ;; (send *ri* :wait-interpolation nil 0.01)
     (send *viewer* :draw-objects)
     )
   (ros::spin-once)
   (ros::sleep)
   )
  )

(defun start-imitation-duka ()
  ;; (send *ri* :speak-jp "学習した動作を実行します")
  (ros::subscribe "/imitation/command" std_msgs::Float64MultiArray #'imitation-cmd-cb)
  ;; (setq *now-cmd* (send *pr2* :rarm :angle-vector))
  (setq *now-cmd* nil)
  (ros::rate 5);;10
  (do-until-key
      (when *now-cmd*
        ;; subscribeして動作する．
        (send *pr2* :head-neck-y :joint-angle 0.0)
        (send *pr2* :head-neck-p :joint-angle 60.0)
        (setq now-rarm-cmd (map float-vector #'(lambda (x) (rad2deg x)) (subseq *now-cmd* 0 7)))
        (setq now-r-gripper-cmd (elt *now-cmd* 7))
        (format t "now rarm cmd ~A ~%" now-rarm-cmd)
        (send *pr2* :rarm :angle-vector now-rarm-cmd)
        ;; (send *ri* :angle-vector (send *pr2* :angle-vector))
        (send *ri* :angle-vector-sequence (list (send *pr2* :angle-vector)) (list 400)) ;;400
        (format t "now r-gripper cmd ~A ~%" now-r-gripper-cmd)
        (if (>= now-r-gripper-cmd 1)
            (send *ri* :start-grasp :rarm)
            (send *ri* :stop-grasp :rarm)
            )

        ;; (send *ri* :wait-interpolation nil 0.01)
        (send *viewer* :draw-objects)
        )
    (ros::spin-once)
    (ros::sleep)
    )
  )

;; use config codes
(defun convert-cmd (num &key
                          (type "revolute")
                          (clamp nil))
  (let (ans)
    (cond
      ((equal type "revolute")
       (print "revolute")
       (setq ans (rad2deg num))
       )
      ((equal type "prismatic")
       (print "prismatic")
       (setq ans (* 1000 num))
       )
      ((equal type "continuous")
       (print "continuous")
       (setq ans (rad2deg num)) ;; todo add clamp?
       )
      (t
       (format t "joint type ~A is not expected~%" type)
       ))
    ans))

(defun initial-pose-config (&key (config-name "config.yaml"))
  (let* ((config-list nil)
         (initial_joints nil)
         (cmd-list nil)
         (l_gripper_flag nil)
         (r_gripper_flag nil)
         )
    (setq config-list (load-config-from-name config-name))
    (setq initial_joints (elt config-list 2))

    (setq cmd-list ())
    (dolist (joint initial_joints)
      (setq now-cmd (elt joint 0))
      (setq now-joint-clamp nil)
      (dolist (elm joint)
        (if (equal (elt elm 0) "name") (setq now-joint-name (cdr elm)))
        (if (equal (elt elm 0) "type") (setq now-joint-type (cdr elm)))
        (if (equal (elt elm 0) "clamp") (setq now-joint-clamp (cdr elm)))
        (if (equal (elt elm 0) "value") (setq now-joint-value (cdr elm)))
        )
      (format t "name: ~A, value: ~A ~%" now-joint-name now-joint-value)
      (setq cmd-list (append cmd-list (list (list now-joint-name (convert-cmd now-joint-value :type now-joint-type :clamp now-joint-clamp)))))
      (when (equal now-joint-name "l_gripper_joint")
        (setq l_gripper_flag t)
        (setq l_gripper_val now-joint-value))
      (when (equal now-joint-name "r_gripper_joint")
        (setq r_gripper_flag t)
        (setq r_gripper_val now-joint-value))
      )

    ;; set cmd to robot
    (dolist (joint (send *pr2* :joint-list))
      (dolist (cmd cmd-list)
        (when (equal (send joint :name) (elt cmd 0))
          (cond
            ((equal (elt cmd 0) "l_gripper_joint") ;; pr2のjoint-listの中にはgripperは無いらしい
             (send *ri* :move-gripper :larm (* 0.001 (elt cmd 1))))
            ((equal (elt cmd 0) "r_gripper_joint")
             (send *ri* :move-gripper :rarm (* 0.001 (elt cmd 1))))
            (t
             (send joint :joint-angle (elt cmd 1)))
            ))))
    (send *pr2* :head-neck-y :joint-angle 0.0)
    (send *pr2* :head-neck-p :joint-angle 60.0)
    (send *ri* :angle-vector (send *pr2* :angle-vector))
    (when l_gripper_flag
      (send *ri* :move-gripper :larm l_gripper_val))
    (when r_gripper_flag
      (send *ri* :move-gripper :rarm r_gripper_val))
    (send *ri* :wait-interpolation)
    ))

(defun start-imitation-config (&key
                                 (config-name "config.yaml")
                                 (g-thre 0.5)
                                 )
  (let* ((target-input nil)
         (ik-target nil)
         (now-cmd nil)
         (larm-cmd nil)
         (l_gripper_flag nil)
         (r_gripper_flag nil)
         ;; (control_joints nil)
         )
    ;; TODO change initial pose
    (setq config-list (load-config-from-name config-name))
    (setq control_joints (elt config-list 0))
    (setq dim_list (elt config-list 1))

    (ros::subscribe "/imitation/command" std_msgs::Float64MultiArray #'imitation-cmd-cb)
    (setq *now-cmd* nil)

    (ros::rate 5);;10
    (do-until-key
        (when *now-cmd*
          ;; subscribeして動作する．
          (send *pr2* :head-neck-y :joint-angle 0.0)
          (send *pr2* :head-neck-p :joint-angle 60.0)

          ;; convert target-input to joint
          (setq cmd-list ())
          (dotimes (i (length control_joints))
            (setq control_joint (elt control_joints i))
            (setq now-joint-clamp nil)
            (dolist (elm control_joint)
              (if (equal (elt elm 0) "name") (setq now-joint-name (cdr elm)))
              (if (equal (elt elm 0) "type") (setq now-joint-type (cdr elm)))
              (if (equal (elt elm 0) "clamp") (setq now-joint-clamp (cdr elm)))
              )
            (format t "now name:~A, type:~A, clamp:~A~%" now-joint-name now-joint-type now-joint-clamp)
            (setq cmd-list (append cmd-list (list (list now-joint-name (convert-cmd (elt *now-cmd* i) :type now-joint-type :clamp now-joint-clamp)))))
            (when (equal now-joint-name "l_gripper_joint")
              (setq l_gripper_flag t)
              )
            (when (equal now-joint-name "r_gripper_joint")
              (setq r_gripper_flag t)
              )
            )
          (format t "now cmd-list ~A ~%" cmd-list)
          ;; set cmd to pr2
          (dolist (joint (send *pr2* :joint-list))
            (dolist (cmd cmd-list)
              (when (equal (send joint :name) (elt cmd 0))
                (cond
                  ((equal (elt cmd 0) "l_gripper_joint") ;; pr2のjoint-listの中にはgripperは無いらしい
                   (send *ri* :move-gripper :larm (* 0.001 (elt cmd 1))))
                  ((equal (elt cmd 0) "r_gripper_joint")
                   (send *ri* :move-gripper :rarm (* 0.001 (elt cmd 1))))
                  (t
                   (send joint :joint-angle (elt cmd 1)))
                  ))))
          ;; (send *ri* :angle-vector-sequence (list (send *pr2* :angle-vector)) (list 600))
          (send *ri* :angle-vector-sequence (list (send *pr2* :angle-vector)) (list 400))
          ;; (send *ri* :angle-vector (send *pr2* :angle-vector))
          (when l_gripper_flag
            (dolist (cmd cmd-list)
              (when (equal "l_gripper_joint" (elt cmd 0))
                (format t "now l_gripper val : ~A ~%" (elt cmd 1))
                (if (< (elt cmd 1) (* 1000 g-thre))
                    (send *ri* :stop-grasp :larm)
                    (send *ri* :start-grasp :larm)))))
          ;; (send *ri* :move-gripper :larm (elt cmd 1)))))
          (when r_gripper_flag
            (dolist (cmd cmd-list)
              (when (equal "r_gripper_joint" (elt cmd 0))
                (format t "now r_gripper val : ~A ~%" (elt cmd 1))
                (if (< (elt cmd 1) (* 1000 g-thre))
                    (send *ri* :stop-grasp :rarm)
                    (send *ri* :start-grasp :rarm)))))
          ;; (send *ri* :move-gripper :rarm (elt cmd 1)))))
          )
      (ros::spin-once)
      (ros::sleep)
      )
    )
  )
