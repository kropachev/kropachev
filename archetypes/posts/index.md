+++
# Основные
title = "{{ replace .Name "-" " " | title }}"
date = {{ .Date }}
lastmod = {{ .Date }}
draft = true

# SEO
description = ""     # короткое описание для карточки
summary = ""         # если нужно задать ьотдельно

# Таксономии
categories = ["short"]
tags = []
series = []
authors = []

# URL и навигация
slug = ""            # обычно не нужно, берется из имени папки
aliases = []         # старый урл для редиректа
weight = 0           # чем меньше, тем выше в списке

# Медиа и оформление
image = "cover.jpg"  # обложка поста (путь относительно папки)
images = []          # список OG:image для соцсетей.
toc = false          # отображение оглавления
readingTime = true
math = false         # формулы KaTeX

# Дополнительно
canonicalURL = ""
robots = "index,follow" # директивы индексации.
+++

Короткий пост...