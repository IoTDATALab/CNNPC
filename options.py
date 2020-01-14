#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # model arguments
    parser.add_argument('--model', type=str, default='resnet', 
                        help='model name: resnet mobilenet')
    parser.add_argument('--end_device', type=str, default='MI8', 
                        help='end device name: MI8 MI8SE')
    parser.add_argument('--bandwidth', type=str, default='10', 
                        help='edge-cloud bandwidth(M): eg. 10 5 1')
    parser.add_argument('--A0', type=float, default=0, 
                        help='Given Accuracy: A0')
    parser.add_argument('--T0', type=float, default=0, 
                        help='Given Latency: T0')     

    args = parser.parse_args()
    return args
