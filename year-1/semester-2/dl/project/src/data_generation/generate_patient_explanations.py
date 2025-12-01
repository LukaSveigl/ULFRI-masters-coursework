# generate_patient_explanations.py
# 
# This file contains a script that generates a dataset of texts from the patient's or doctor's perspectives, 
# describing potential symptoms. The script uses predefined templates for sentences that describe one, two, 
# three or more symptoms. It also has a provision for handling multiple symptoms.
# 
# The templates are filled with symptom names to generate sentences that sound natural and are varied in structure. 
# This is done to mimic the variety of ways patients or doctors might describe symptoms in real-world scenarios.
# 
# The generated dataset can be used for training natural language processing models to understand and extract 
# symptom information from patient or doctor narratives. This is crucial for automated disease diagnosis systems 
# that rely on understanding patient symptoms.
import re
import json
import pandas as pd
import random
from nltk.corpus import wordnet
from textattack.transformations import WordSwapWordNet
from textattack.transformations import WordSwapRandomCharacterDeletion
from textattack.transformations import WordSwapRandomCharacterInsertion
from textattack.transformations import WordSwapRandomCharacterSubstitution
from textattack.transformations import WordSwapNeighboringCharacterSwap
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.shared import AttackedText
from transformers import AutoTokenizer, AutoModelForSequenceClassification

single_symptom_templates = [
    "The patient is experiencing {symptom1}.",
    "{symptom1} has been reported by the patient.",
    "Patient complains of {symptom1}.",
    "Symptoms reported by the patient include {symptom1}.",
    "Doctor, I've been experiencing this persistent {symptom1} for over a week now. It's really starting to worry me because it doesn't seem to be getting any better. I’ve tried some over-the-counter treatments, but nothing seems to help. It's affecting my daily activities and causing a lot of discomfort.",
    "I've noticed a troubling {symptom1} recently.",
    "I've been having this persistent {symptom1} that I can't seem to shake off.",
    "The patient reports a constant {symptom1}.",
    "{symptom1} has been a major concern for the patient.",
    "I've been dealing with {symptom1} for a few days now.",
    "Lately, I've been experiencing {symptom1} and it's quite bothersome.",
    "I've been struggling with {symptom1} lately.",
    "Patient mentions experiencing {symptom1} continuously.",
    "The main issue for the patient is {symptom1}.",
    "I've been facing issues with {symptom1}.",
    "The patient has been troubled by {symptom1}.",
    "The occurrence of {symptom1} has been noted by the patient.",
    "There's this {symptom1} that's been bothering me.",
    "Experiencing {symptom1} has been a significant issue for me.",
    "The patient has consistently noted {symptom1}.",
    "I'm concerned about this ongoing {symptom1}.",
    "Patient consistently reports {symptom1}.",
    "I've been dealing with an uncomfortable {symptom1}.",
    "There's a persistent {symptom1} that I've been experiencing.",
    "This {symptom1} is really troubling me."
]

two_symptom_templates = [
    "The patient is suffering from {symptom1} and {symptom2}.",
    "The patient has reported {symptom1} and {symptom2}.",
    "{symptom1} and {symptom2} have been reported by the patient.",
    "Patient complains of {symptom1} and {symptom2}.",
    "Symptoms reported by the patient include {symptom1} and {symptom2}.",
    "The patient is experiencing {symptom1} and {symptom2}.",
    "The patient has {symptom1} and {symptom2}.",
    "Both {symptom1} and {symptom2} are present.",
    "I've been having {symptom1} and {symptom2} lately.",
    "The occurrence of {symptom1} and {symptom2} has been noted by the patient.",
    "I've been dealing with {symptom1} and {symptom2} simultaneously.",
    "The patient mentions experiencing both {symptom1} and {symptom2}.",
    "There's been a combination of {symptom1} and {symptom2}.",
    "Experiencing {symptom1} along with {symptom2} has been troubling me.",
    "The patient reports {symptom1} as well as {symptom2}.",
    "I've noticed a mix of {symptom1} and {symptom2}.",
    "Patient consistently reports both {symptom1} and {symptom2}.",
    "These symptoms, {symptom1} and {symptom2}, are affecting the patient.",
    "I've been troubled by {symptom1} and {symptom2}.",
    "There's an ongoing issue with {symptom1} and {symptom2}.",
    "Both {symptom1} and {symptom2} have been causing discomfort.",
    "I’m experiencing a persistent {symptom1} alongside {symptom2}.",
    "These issues, {symptom1} and {symptom2}, have been persistent.",
    "I've been struggling with {symptom1} and {symptom2} for some time.",
    "The patient mentions a consistent problem with {symptom1} and {symptom2}."
]

three_symptom_templates = [
    "The patient describes symptoms of {symptom1}, {symptom2}, and {symptom3}.",
    "The patient is experiencing {symptom1}, {symptom2}, and {symptom3}.",
    "The patient has reported {symptom1}, {symptom2}, and {symptom3}.",
    "{symptom1}, {symptom2}, and {symptom3} have been reported by the patient.",
    "{symptom1}, {symptom2}, and {symptom3} have been observed by the patient.",
    "Patient complains of {symptom1}, {symptom2}, and {symptom3}.",
    "Symptoms reported by the patient include {symptom1}, {symptom2}, and {symptom3}.",
    "I’ve been dealing with {symptom1}, {symptom2}, and {symptom3} lately.",
    "There's a combination of {symptom1}, {symptom2}, and {symptom3} troubling the patient.",
    "Patient has noted the presence of {symptom1}, {symptom2}, and {symptom3}.",
    "Experiencing {symptom1}, {symptom2}, and {symptom3} has been a challenge.",
    "The patient mentions {symptom1}, {symptom2}, and {symptom3} continuously.",
    "All three symptoms, {symptom1}, {symptom2}, and {symptom3}, are present.",
    "I've been having trouble with {symptom1}, {symptom2}, and {symptom3}.",
    "The patient describes issues with {symptom1}, {symptom2}, and {symptom3}.",
    "I'm struggling with {symptom1}, {symptom2}, and {symptom3} simultaneously.",
    "The patient consistently reports {symptom1}, {symptom2}, and {symptom3}.",
    "I've noticed a mix of {symptom1}, {symptom2}, and {symptom3}.",
    "The occurrence of {symptom1}, {symptom2}, and {symptom3} has been noted.",
    "I've been dealing with {symptom1}, {symptom2}, and {symptom3} for some time.",
    "There's an ongoing issue with {symptom1}, {symptom2}, and {symptom3}.",
    "All these symptoms, {symptom1}, {symptom2}, and {symptom3}, have been persistent.",
    "I'm experiencing a persistent {symptom1}, along with {symptom2} and {symptom3}.",
    "These issues, {symptom1}, {symptom2}, and {symptom3}, have been causing discomfort.",
    "The patient mentions continuous problems with {symptom1}, {symptom2}, and {symptom3}."

]

multi_symptom_templates = [
    "The patient is suffering from multiple symptoms including {symptom1}, {symptom2}, {symptom3}, and {symptom4}.",
    "{symptom1}, {symptom2}, {symptom3}, and {symptom4} have been documented in the patient's record.",
    "Symptoms such as {symptom1}, {symptom2}, {symptom3}, and {symptom4} are affecting the patient.",
    "A combination of {symptom1}, {symptom2}, {symptom3}, and {symptom4} is reported.",
    "The patient has been experiencing several symptoms, notably {symptom1}, {symptom2}, {symptom3}, and {symptom4}.",
    "There's an array of symptoms, including {symptom1}, {symptom2}, {symptom3}, and {symptom4}, affecting the patient.",
    "Patient is dealing with {symptom1}, {symptom2}, {symptom3}, and {symptom4}.",
    "Among the reported symptoms are {symptom1}, {symptom2}, {symptom3}, and {symptom4}.",
    "The patient describes a series of symptoms: {symptom1}, {symptom2}, {symptom3}, and {symptom4}.",
    "There have been multiple symptoms observed: {symptom1}, {symptom2}, {symptom3}, and {symptom4}.",
    "Patient complains of {symptom1}, {symptom2}, {symptom3}, and {symptom4}.",
    "Experiencing {symptom1}, {symptom2}, {symptom3}, and {symptom4} has been very challenging.",
    "The patient mentions continuous issues with {symptom1}, {symptom2}, {symptom3}, and {symptom4}.",
    "All these symptoms, {symptom1}, {symptom2}, {symptom3}, and {symptom4}, have been persistent.",
    "I've been having trouble with {symptom1}, {symptom2}, {symptom3}, and {symptom4}.",
    "The patient has noted the presence of {symptom1}, {symptom2}, {symptom3}, and {symptom4}.",
    "I've been dealing with {symptom1}, {symptom2}, {symptom3}, and {symptom4} simultaneously.",
    "There's an ongoing issue with {symptom1}, {symptom2}, {symptom3}, and {symptom4}.",
    "The occurrence of {symptom1}, {symptom2}, {symptom3}, and {symptom4} has been troubling the patient.",
    "These issues, {symptom1}, {symptom2}, {symptom3}, and {symptom4}, have been persistent.",
    "The patient reports consistent problems with {symptom1}, {symptom2}, {symptom3}, and {symptom4}.",
    "I'm struggling with {symptom1}, {symptom2}, {symptom3}, and {symptom4}.",
    "Patient consistently reports {symptom1}, {symptom2}, {symptom3}, and {symptom4}.",
    "These symptoms, {symptom1}, {symptom2}, {symptom3}, and {symptom4}, are affecting me greatly.",
    "I've noticed a troubling combination of {symptom1}, {symptom2}, {symptom3}, and {symptom4}."

]

general_templates = [
    "The patient is experiencing a variety of symptoms including {symptom1}, {symptom2}, and {symptom3}.",
    "Patient complains of several symptoms, notably {symptom1}, {symptom2}, and {symptom3}.",
    "Symptoms reported by the patient include {symptom1}, {symptom2}, and {symptom3}.",
    "I've been struggling with multiple symptoms, such as {symptom1}, {symptom2}, and {symptom3}.",
    "The patient describes experiencing {symptom1}, {symptom2}, and {symptom3}.",
    "There is a combination of {symptom1}, {symptom2}, and {symptom3} affecting the patient.",
    "Among the symptoms reported are {symptom1}, {symptom2}, and {symptom3}.",
    "The patient has been facing issues with {symptom1}, {symptom2}, and {symptom3}.",
    "Experiencing {symptom1}, {symptom2}, and {symptom3} has been very troubling.",
    "The patient mentions continuous problems with {symptom1}, {symptom2}, and {symptom3}.",
    "All these symptoms, {symptom1}, {symptom2}, and {symptom3}, have been persistent.",
    "I've been having trouble with {symptom1}, {symptom2}, and {symptom3}.",
    "The patient has noted the presence of {symptom1}, {symptom2}, and {symptom3}.",
    "I've been dealing with {symptom1}, {symptom2}, and {symptom3} simultaneously.",
    "There's an ongoing issue with {symptom1}, {symptom2}, and {symptom3}.",
    "The occurrence of {symptom1}, {symptom2}, and {symptom3} has been troubling the patient.",
    "These issues, {symptom1}, {symptom2}, and {symptom3}, have been persistent.",
    "The patient reports consistent problems with {symptom1}, {symptom2}, and {symptom3}.",
    "I'm struggling with {symptom1}, {symptom2}, and {symptom3}.",
    "Patient consistently reports {symptom1}, {symptom2}, and {symptom3}.",
    "These symptoms, {symptom1}, {symptom2}, and {symptom3}, are affecting me greatly.",
    "I've noticed a troubling combination of {symptom1}, {symptom2}, and {symptom3}.",
    "The patient is experiencing a series of symptoms including {symptom1}, {symptom2}, and {symptom3}.",
    "Patient describes issues with {symptom1}, {symptom2}, and {symptom3}."
]

in_depth_templates = [
    "You see, Doctor, I've been experiencing {symptom1} for about a week now. It's been quite persistent and is now accompanied by {symptom2}. I thought it might go away, but instead, {symptom3} has started. I'm also feeling {symptom4}, which is adding to my discomfort. I need to understand what's going on with my body.",
    "I've been having a really hard time with {symptom1}. Just when I thought it couldn't get worse, I started experiencing {symptom2}. Then, {symptom3} showed up, and now I'm also dealing with {symptom4}. It's affecting my work, my sleep, and my overall well-being.",
    "Doctor, my main concern started with {symptom1}, but soon after, I noticed {symptom2}. Things took a turn for the worse when {symptom3} began. Now, I'm also contending with {symptom4}. It's a combination that's really taking a toll on me.",
    "I've been dealing with {symptom1} and {symptom2} for the past few days. It started with just {symptom1}, but then I noticed {symptom2} as well. These symptoms are making it hard for me to focus on work and get through my daily routine. I've been trying to rest and stay hydrated, but they just won't go away.",
    "I've noticed {symptom1}, {symptom2}, and {symptom3} recently. At first, I thought it was just a minor issue, but now all three symptoms are persistent and worsening. I can't sleep properly, and my appetite has decreased significantly. I’m really concerned because it's affecting my overall well-being and productivity.",
    "I'm really struggling with {symptom1}, {symptom2}, {symptom3}, and {symptom4}. It's like my body is falling apart. I can't perform my daily tasks, and I'm constantly exhausted. I've tried different remedies, but nothing seems to work. The combination of these symptoms is overwhelming, and I'm not sure what to do.",
    "Doctor, I've been having this persistent {symptom1} for about two weeks, and just recently, {symptom2} has started. Along with these, I've also noticed {symptom3}. It's really starting to wear me down because I can't find any relief. I’ve been taking some medication, but it doesn't seem to be helping. Additionally, {symptom4} began a few days ago, adding to my discomfort. I need to understand what's going on with my body and find a way to manage these symptoms.",
    "You see, Doctor, my health has been deteriorating over the past month. It started with a persistent {symptom1}, which I initially ignored, thinking it would go away. However, after a week, I began experiencing {symptom2} as well. This really worried me, but I hoped it was just a temporary issue. Unfortunately, things took a turn for the worse when {symptom3} started. Now, I'm also contending with {symptom4}. This combination of symptoms is not only causing physical discomfort but also affecting my mental health. I’m constantly anxious and stressed about my condition, which is impacting my work and personal life. I've tried various treatments and home remedies, but nothing seems to work. I need your help to diagnose and treat these symptoms effectively.",
    "Doctor, my main concern started with {symptom1}, which has been bothering me for several weeks. I initially thought it was just a minor issue, but soon after, I noticed {symptom2}. I tried to manage these symptoms on my own, but then {symptom3} began, adding to my worries. Just when I thought it couldn't get worse, {symptom4} appeared. I’ve been feeling extremely overwhelmed and unable to focus on anything else. The pain and discomfort are constant, making it hard for me to perform daily activities. I've been researching my symptoms online, but it's only making me more anxious. I really need your expertise to figure out what's going on and how to treat these symptoms. I'm desperate for some relief and a return to my normal life.",
    "Doctor, over the past month, I've been experiencing a range of symptoms that are really affecting my quality of life. It all started with {symptom1}, which I didn’t think much of initially. However, as the days went by, I began to notice {symptom2}. I tried to ignore it, hoping it would go away on its own, but it persisted. A couple of weeks ago, {symptom3} started, and that’s when I became really concerned. I began taking some over-the-counter medications, but they didn’t provide much relief. Recently, {symptom4} has also developed, making the situation even worse. I'm finding it increasingly difficult to carry out my daily activities, and my productivity at work has plummeted. I'm constantly tired, irritable, and worried about my health. I've tried various home remedies and even consulted Dr. Google, but nothing seems to help. I need your professional advice to understand what’s happening to me and how to get back to my normal life.",
    "Doctor, I've been feeling really unwell for the past few weeks, and it's been getting progressively worse. It all began with a nagging {symptom1} that wouldn't go away. At first, I thought it was something minor, but it persisted and soon I started experiencing {symptom2} as well. This combination made it difficult to go about my daily activities. To make matters worse, about a week ago, I noticed {symptom3}, which has been incredibly painful and debilitating. Just a few days ago, I also started dealing with {symptom4}, and it's made me feel completely overwhelmed. These symptoms are affecting every aspect of my life, my work, my sleep, and even my mood. I'm constantly anxious about what could be causing all these issues and really need your help to figure out what's going on and how to get better.",
    "Lately, I've been feeling really off, and it seems like my body is just falling apart. It all started with a persistent {symptom1}, which I tried to ignore, hoping it would go away on its own. Unfortunately, it didn't, and soon after, I began to experience {symptom2}. This made my situation much worse and started to interfere with my daily routine. Then, out of nowhere, {symptom3} appeared, making everything even more challenging. Just when I thought I couldn't handle any more, I began to notice {symptom4}. These symptoms have been so overwhelming that I find it hard to focus on anything else. I'm really scared about what's happening to me and need your expertise to understand what might be causing all these problems.",
    "Doctor, I'm at my wit's end with these symptoms. It all began with a mild {symptom1}, which I thought was nothing to worry about. However, it persisted and started to interfere with my daily life. Then, I started experiencing {symptom2}, which made me feel even worse. As if that wasn't enough, {symptom3} appeared, and it's been incredibly painful. Recently, I've also developed {symptom4}, which has made my situation even more unbearable. These symptoms are affecting my work, my sleep, and my overall well-being. I've tried everything I can think of to manage them, but nothing seems to help. I'm really worried about my health and need your help to figure out what's going on and how to get better.",
    "I've been feeling really unwell for the past few weeks, and it's been a constant struggle. It started with a persistent {symptom1}, which I hoped would go away on its own. Instead, it got worse, and I started experiencing {symptom2} shortly after. This made it difficult to go about my daily activities. Then, about a week ago, I noticed {symptom3}, which has been incredibly painful and debilitating. Just a few days ago, I also started dealing with {symptom4}, and it's made me feel completely overwhelmed. These symptoms are affecting every aspect of my life, my work, my sleep, and even my mood. I'm constantly anxious about what could be causing all these issues and really need your help to figure out what's going on and how to get better.",
    "Doctor, I'm here because I've been experiencing a range of symptoms that are making my life very difficult. It started with a mild {symptom1}, which I didn't think much of at first. However, it persisted and soon I started experiencing {symptom2} as well. This combination made it difficult to go about my daily activities. To make matters worse, about a week ago, I noticed {symptom3}, which has been incredibly painful and debilitating. Just a few days ago, I also started dealing with {symptom4}, and it's made me feel completely overwhelmed. These symptoms are affecting every aspect of my life, my work, my sleep, and even my mood. I'm constantly anxious about what could be causing all these issues and really need your help to figure out what's going on and how to get better.",
    "Doctor, I've been struggling with several symptoms that are really affecting my quality of life. It all started with a persistent {symptom1}, which I thought would go away on its own. Instead, it got worse, and I started experiencing {symptom2} shortly after. This made it difficult to go about my daily activities. Then, about a week ago, I noticed {symptom3}, which has been incredibly painful and debilitating. Just a few days ago, I also started dealing with {symptom4}, and it's made me feel completely overwhelmed. These symptoms are affecting every aspect of my life, my work, my sleep, and even my mood. I'm constantly anxious about what could be causing all these issues and really need your help to figure out what's going on and how to get better.",
    "I've been feeling really off for the past few weeks, and it's been a constant struggle. It started with a persistent {symptom1}, which I hoped would go away on its own. Instead, it got worse, and I started experiencing {symptom2} shortly after. This made it difficult to go about my daily activities. Then, about a week ago, I noticed {symptom3}, which has been incredibly painful and debilitating. Just a few days ago, I also started dealing with {symptom4}, and it's made me feel completely overwhelmed. These symptoms are affecting every aspect of my life, my work, my sleep, and even my mood. I'm constantly anxious about what could be causing all these issues and really need your help to figure out what's going on and how to get better.",
    "Doctor, I've been having a really tough time with my health lately. It all began with a mild {symptom1}, which I didn't think much of at first. However, it persisted and soon I started experiencing {symptom2} as well. This combination made it difficult to go about my daily activities. To make matters worse, about a week ago, I noticed {symptom3}, which has been incredibly painful and debilitating. Just a few days ago, I also started dealing with {symptom4}, and it's made me feel completely overwhelmed. These symptoms are affecting every aspect of my life, my work, my sleep, and even my mood. I'm constantly anxious about what could be causing all these issues and really need your help to figure out what's going on and how to get better.",
    "I've been feeling really unwell lately, and it's been getting progressively worse. It started with a persistent {symptom1}, which I thought would go away on its own. Instead, it got worse, and I started experiencing {symptom2} shortly after. This made it difficult to go about my daily activities. Then, about a week ago, I noticed {symptom3}, which has been incredibly painful and debilitating. Just a few days ago, I also started dealing with {symptom4}, and it's made me feel completely overwhelmed. These symptoms are affecting every aspect of my life, my work, my sleep, and even my mood. I'm constantly anxious about what could be causing all these issues and really need your help to figure out what's going on and how to get better.",
    "Doctor, I'm really struggling with my health these days. It all started with a mild {symptom1}, which I thought would go away on its own. However, it persisted and soon I started experiencing {symptom2} as well. This combination made it difficult to go about my daily activities. To make matters worse, about a week ago, I noticed {symptom3}, which has been incredibly painful and debilitating. Just a few days ago, I also started dealing with {symptom4}, and it's made me feel completely overwhelmed. These symptoms are affecting every aspect of my life, my work, my sleep, and even my mood. I'm constantly anxious about what could be causing all these issues and really need your help to figure out what's going on and how to get better.",
    "I've been feeling really off lately, and it's been a constant struggle. It started with a persistent {symptom1}, which I thought would go away on its own. Instead, it got worse, and I started experiencing {symptom2} shortly after. This made it difficult to go about my daily activities. Then, about a week ago, I noticed {symptom3}, which has been incredibly painful and debilitating. Just a few days ago, I also started dealing with {symptom4}, and it's made me feel completely overwhelmed. These symptoms are affecting every aspect of my life, my work, my sleep, and even my mood. I'm constantly anxious about what could be causing all these issues and really need your help to figure out what's going on and how to get better.",
    "Doctor, I've been feeling terrible lately, and it's been affecting every part of my life. It all started with a persistent {symptom1} that I thought was just a minor issue. But it didn't go away and instead, it became worse. Shortly after, I started experiencing {symptom2}, which added to my discomfort. Then, I noticed {symptom3}, which was completely unexpected and quite painful. To top it off, a few days ago, I developed {symptom4}. These symptoms have made it extremely difficult to work, sleep, and even relax. I'm constantly worried about my health, and it's affecting my mental state as well. I need to understand what's happening to me and how I can get some relief from these symptoms.",
    "I've been going through a really tough time with my health recently. It began with a mild {symptom1}, which I thought would resolve on its own. However, it persisted and soon I started experiencing {symptom2} as well. This combination has been really challenging to deal with. Then, a week ago, I noticed {symptom3}, which added a lot of pain to my daily life. Just a few days ago, I started having {symptom4}, and now I'm feeling completely overwhelmed. These symptoms are affecting my work, my sleep, and my overall mood. I'm really worried about what's causing all these issues and need your help to find a solution.",
    "Doctor, I've been dealing with several symptoms that are making life very difficult. It started with a constant {symptom1}, which I hoped would go away. Instead, it got worse and soon I began experiencing {symptom2}. This new symptom made things even harder. Then, about a week ago, I developed {symptom3}, which has been incredibly painful. Recently, I've also started experiencing {symptom4}. These symptoms are affecting my ability to function normally. I've been feeling extremely anxious about my health and need your guidance to understand what's happening and how to get better.",
    "I've been feeling really unwell for the past few weeks, and it's been a struggle to get through each day. It all started with a persistent {symptom1}, which was annoying but manageable at first. However, it didn't go away and soon I began to experience {symptom2} as well. This made my situation much worse and started to interfere with my daily routine. Then, about a week ago, I noticed {symptom3}, which has been incredibly painful and debilitating. Just a few days ago, I also started dealing with {symptom4}, and it's made me feel completely overwhelmed. These symptoms are affecting every aspect of my life, my work, my sleep, and even my mood. I'm constantly anxious about what could be causing all these issues and really need your help to figure out what's going on and how to get better.",
    "Doctor, I'm here because I've been experiencing a range of symptoms that are making my life very difficult. It started with a mild {symptom1}, which I didn't think much of at first. However, it persisted and soon I started experiencing {symptom2} as well. This combination made it difficult to go about my daily activities. To make matters worse, about a week ago, I noticed {symptom3}, which has been incredibly painful and debilitating. Just a few days ago, I also started dealing with {symptom4}, and it's made me feel completely overwhelmed. These symptoms are affecting every aspect of my life, my work, my sleep, and even my mood. I'm constantly anxious about what could be causing all these issues and really need your help to figure out what's going on and how to get better.",
    "Doctor, I've been struggling with several symptoms that are really affecting my quality of life. It all started with a persistent {symptom1}, which I thought would go away on its own. Instead, it got worse, and I started experiencing {symptom2} shortly after. This made it difficult to go about my daily activities. Then, about a week ago, I noticed {symptom3}, which has been incredibly painful and debilitating. Just a few days ago, I also started dealing with {symptom4}, and it's made me feel completely overwhelmed. These symptoms are affecting every aspect of my life, my work, my sleep, and even my mood. I'm constantly anxious about what could be causing all these issues and really need your help to figure out what's going on and how to get better.",
    "Doctor, I've been having a really tough time with my health lately. It all began with a mild {symptom1}, which I didn't think much of at first. However, it persisted and soon I started experiencing {symptom2} as well. This combination made it difficult to go about my daily activities. To make matters worse, about a week ago, I noticed {symptom3}, which has been incredibly painful and debilitating. Just a few days ago, I also started dealing with {symptom4}, and it's made me feel completely overwhelmed. These symptoms are affecting every aspect of my life, my work, my sleep, and even my mood. I'm constantly anxious about what could be causing all these issues and really need your help to figure out what's going on and how to get better.",
    "Doctor, I've been feeling quite unwell lately, and it's been affecting every part of my life. It all started with a persistent {symptom1} that I just couldn't shake off. Initially, I thought it was something minor and that it would go away on its own, but it has persisted for several weeks now. On top of that, I've also begun to experience {symptom2}, which is really adding to my discomfort. It's been making my daily activities much more challenging. Just when I thought it couldn't get worse, I started noticing {symptom3}. It's not just physical discomfort; it's starting to affect my mental well-being too. To make matters worse, in the past few days, I've also developed {symptom4}. I feel like my health is rapidly declining, and I'm becoming increasingly anxious about these symptoms. They seem to be interconnected, and I'm worried that something serious might be going on. Could you please help me understand what's happening?",
    "Doctor, I'm really struggling with my health these days. It all started with a mild {symptom1}, which I thought would go away on its own. However, it persisted and soon I started experiencing {symptom2} as well. This combination made it difficult to go about my daily activities. To make matters worse, about a week ago, I noticed {symptom3}, which has been incredibly painful and debilitating. Just a few days ago, I also started dealing with {symptom4}, and it's made me feel completely overwhelmed. These symptoms are affecting every aspect of my life, my work, my sleep, and even my mood. I'm constantly anxious about what could be causing all these issues and really need your help to figure out what's going on and how to get better.",
    "Doctor, I've been going through a really tough time with my health recently. It began with a mild {symptom1}, which I thought would resolve on its own. However, it persisted and soon I started experiencing {symptom2} as well. This combination has been really challenging to deal with. Then, a week ago, I noticed {symptom3}, which added a lot of pain to my daily life. Just a few days ago, I started having {symptom4}, and now I'm feeling completely overwhelmed. These symptoms are affecting my work, my sleep, and my overall mood. I'm really worried about what's causing all these issues and need your help to find a solution.",
    "Doctor, I've been dealing with a persistent {symptom1}. It worsened over time and was joined by {symptom2}. Recently, {symptom3} started affecting me, and now I've also noticed {symptom4}. I’m struggling to manage these symptoms and need your help.",
    "I initially experienced {symptom1}, which has now been accompanied by {symptom2}. Over the past few days, {symptom3} has started, and {symptom4} emerged. It’s been very challenging to handle these issues.",
    "I began with {symptom1} a couple of weeks ago. Then, {symptom2} appeared. Soon after, {symptom3} followed, and just recently, {symptom4} added to my discomfort. I'm at a loss on how to proceed.",
    "Doctor, my condition started with {symptom1}. Then {symptom2} began affecting me. As if that wasn’t enough, {symptom3} appeared, and lately, {symptom4} has surfaced. These symptoms are greatly impacting my life.",
    "Initially, it was just {symptom1}. Then {symptom2} started troubling me. Now, I’m also dealing with {symptom3} and {symptom4}. These combined symptoms are overwhelming and I need your expertise.",
    "I’ve been trying to manage a persistent {symptom1}. After a week, {symptom2} showed up. Recently, {symptom3} has started to bother me, and just yesterday, {symptom4} appeared. It’s affecting my ability to function normally.",
    "It started with {symptom1} a few weeks ago. Then I noticed {symptom2}. Shortly after, {symptom3} began, and now {symptom4} is present. The severity of these symptoms is increasing.",
    "I’ve had {symptom1} for a while now, but recently {symptom2} began. Then, {symptom3} showed up, and now {symptom4} is affecting me. I'm finding it hard to cope with these multiple symptoms.",
    "Doctor, first I had {symptom1}, which was bad enough. Then came {symptom2}. Recently, I developed {symptom3}, and just a few days ago, {symptom4} emerged. These symptoms are making life very difficult.",
    "I've been experiencing {symptom1} for a few weeks. Not long after, {symptom2} appeared. Recently, {symptom3} started affecting me, and just a few days ago, {symptom4} added to my challenges. I need your assistance to manage these symptoms effectively.",
    "I've been dealing with {symptom1} for some time now. Then, {symptom2} showed up. After a while, I noticed {symptom3}, and now, {symptom4} has started as well. I'm having a hard time managing all these symptoms together.",
    "Doctor, I've been facing a tough time with {symptom1}. It was soon followed by {symptom2}. Then, {symptom3} began to trouble me. Just recently, {symptom4} added to my worries. I'm finding it increasingly difficult to manage these issues.",
    "First, it was just {symptom1}. But soon, {symptom2} started. Not long after, {symptom3} appeared. Now, {symptom4} is also present. Handling all these symptoms is becoming unbearable.",
    "It started with {symptom1}. Then, {symptom2} showed up. After some time, {symptom3} began to trouble me. And now, {symptom4} has started as well. I need help managing all these symptoms.",
    "I've been struggling with {symptom1} initially. Soon, {symptom2} appeared. A few days later, {symptom3} started. Recently, {symptom4} also added to my discomfort. These symptoms are making it hard for me to function properly.",
    "At first, there was {symptom1}. Shortly after, {symptom2} began. Then, {symptom3} showed up. Now, {symptom4} has started to affect me. It's becoming overwhelming to deal with all these symptoms.",
    "I've been having issues with {symptom1}. After a while, {symptom2} began. Recently, {symptom3} started. Just a few days ago, {symptom4} appeared. These symptoms are significantly impacting my daily life.",
    "It all started with {symptom1}. Then came {symptom2}. Not long after, {symptom3} began to affect me. Just recently, {symptom4} appeared. Handling all these symptoms is becoming too much.",
    "I initially noticed {symptom1}. Then, {symptom2} appeared. Shortly after, {symptom3} began troubling me. Now, {symptom4} is also present. Managing these symptoms is increasingly difficult.",
    "First, I experienced {symptom1}. After a while, {symptom2} showed up. Then, {symptom3} began. Recently, {symptom4} started as well. Dealing with all these symptoms is quite challenging.",
    "I've been dealing with {symptom1}. Then, {symptom2} appeared. Soon after, {symptom3} started. Now, {symptom4} is present as well. These symptoms are making life very difficult.",
    "It started with {symptom1}. Then, {symptom2} began. After a while, {symptom3} showed up. Now, {symptom4} has also appeared. Handling all these symptoms is becoming unbearable.",
    "I've been struggling with {symptom1}. Shortly after, {symptom2} started. Recently, {symptom3} appeared. Now, {symptom4} has added to my challenges. Managing these symptoms is increasingly difficult.",
    "Initially, there was {symptom1}. Then, {symptom2} began. Not long after, {symptom3} showed up. Recently, {symptom4} appeared. These combined symptoms are making life hard."

]

def count_placeholders(template: str) -> int:
    """
    Count the number of placeholders in a template.

    Args:
        template (str): The template string.
    """
    pattern = re.compile(r'\{symptom\d+\}')
    placeholders = pattern.findall(template)
    return len(placeholders)


word_swap = WordSwapWordNet()
word_swap_deletion = WordSwapRandomCharacterInsertion()
word_swap_insertion = WordSwapRandomCharacterDeletion()
word_swap_substitution = WordSwapRandomCharacterSubstitution()
word_swap_shuffle = WordSwapNeighboringCharacterSwap()


def augment(template: str, model_wrapper: HuggingFaceModelWrapper):
    """
    Augment a template using TextAttack, by replacing, swapping, deleting and inserting words.
    
    Arg: 
        template (str): The template to augment.
    
    Returns:
        str: The augmented template.
    """
    attacked_text = AttackedText(template)
    augmented_texts = word_swap(attacked_text)
    # Select the first transformed text.
    augmented_text = augmented_texts[0] if augmented_texts else attacked_text
    augmented_texts = word_swap_deletion(augmented_text)
    augmented_text = augmented_texts[0] if augmented_texts else augmented_text
    augmented_texts = word_swap_insertion(augmented_text)
    augmented_text = augmented_texts[0] if augmented_texts else augmented_text
    augmented_texts = word_swap_substitution(augmented_text)
    augmented_text = augmented_texts[0] if augmented_texts else augmented_text
    augmented_texts = word_swap_shuffle(augmented_text)
    augmented_text = augmented_texts[0] if augmented_texts else augmented_text
    return augmented_text.text


if __name__ == '__main__':
    training_dataset = pd.read_csv('../data/disease_prediction/Training.csv')

    mode = 'simple' #'iob' # 'simple'

    # Get the symptoms from the training dataset. The symptoms are all but the last
    # column names in the dataset.
    symptoms = training_dataset.columns[:-1]
    symptoms_clean = [symptom.replace('_', ' ') for symptom in symptoms]

    # Combine all templates.
    all_templates = single_symptom_templates + two_symptom_templates + three_symptom_templates + multi_symptom_templates + general_templates + in_depth_templates

    # Keep track of which symptoms have been chosen and how many times they have been chosen.
    chosen_symptoms = {symptom: 0 for symptom in symptoms_clean}

    model = AutoModelForSequenceClassification.from_pretrained('textattack/bert-base-uncased-imdb')
    tokenizer = AutoTokenizer.from_pretrained('textattack/bert-base-uncased-imdb')
    model_wrapper = HuggingFaceModelWrapper(model, tokenizer)

    file = open(f'../data/patient_explanations/patient_explanations_{mode}.json', 'w', encoding='utf-8')
    for _ in range(10000): # range(10000):
        # Select a random template.
        template = random.choice(all_templates)

        num_placeholders = count_placeholders(template)

        # Select random symptoms to fill the placeholders.
        selected_symptoms = random.sample(symptoms_clean, num_placeholders)

        for symptom in selected_symptoms:
            chosen_symptoms[symptom] += 1

        #template = augment(template, model_wrapper)

        split_template = template.split(' ')
        actual_template = []
        template_labels = []
        symptom_counter = 0

        if mode == 'iob':
            # If the mode is IoB (Inside-outside-beginning), label each token in the template seperately.
            for word in split_template:
                if '{' in word:
                    selected_symptom_split = selected_symptoms[symptom_counter].strip().split(' ')
                    for i, symptom_word in enumerate(selected_symptom_split):
                        if ',' in symptom_word:
                            actual_template.append(f'{symptom_word},')
                        elif '.' in symptom_word:
                            actual_template.append(f'{symptom_word}.')
                        else:
                            actual_template.append(f'{symptom_word}')
                        # Label the first word of the symptom as B-SYMPTOM and the rest as I-SYMPTOM.
                        if i == 0:
                            template_labels.append('B-SYMPTOM')
                        else:
                            template_labels.append('I-SYMPTOM')
                    symptom_counter += 1
                else:
                    actual_template.append(f'{word}')
                    template_labels.append('O')
        else:
            # If the mode is simple, the labels are just the list of symptoms. If the 
            # current processed word is not a symptom, just append it to the actual template. 
            # Otherwise, append the symptom and its label to the template labels.
            for word in split_template:
                if '{' in word:
                    selected_symptom = selected_symptoms[symptom_counter]
                    actual_template.append(selected_symptom)
                    template_labels.append(selected_symptom.replace(' ', '_'))
                    symptom_counter += 1
                else:
                    actual_template.append(f'{word}')

        file.write(json.dumps({'text': ' '.join(actual_template).replace('  ', ' '), 'labels': template_labels}) + '\n')
    
    file.close()

    # Print how many symptoms have non-zero counts.
    print(f'Symptoms chosen: {len([symptom for symptom in chosen_symptoms if chosen_symptoms[symptom] > 0])}')
