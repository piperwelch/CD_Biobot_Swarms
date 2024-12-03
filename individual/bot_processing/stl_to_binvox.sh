#!/bin/bash

BOT_ID='Run8bot9'
ALPHA=12

DIR=$HOME"/research/anthrobots/anthrobot-sim"

binvox $DIR"/binvox/$BOT_ID/body_tform_alpha${ALPHA}_pts450000.stl"

binvox $DIR"/binvox/$BOT_ID/cilia_tform.stl"