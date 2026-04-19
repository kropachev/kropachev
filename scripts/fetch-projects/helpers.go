package main

import (
    "bufio"
    "fmt"
    "regexp"
    "strings"
)

var slugRe = regexp.MustCompile(`[^a-zA-Z0-9._-]`)

func parseRepoURL(repoURL string) (owner, name string, err error) {
    trimmed := strings.TrimSuffix(repoURL, "/")
    parts := strings.Split(trimmed, "/")
    if len(parts) < 2 {
        return "", "", fmt.Errorf("invalid repo url: %s", repoURL)
    }
    return parts[len(parts)-2], parts[len(parts)-1], nil
}

func escapeYAML(s string) string {
    s = strings.ReplaceAll(s, `\`, `\\`)
    s = strings.ReplaceAll(s, `"`, `\"`)
    return s
}

func sanitizeSlug(s string) string {
    s = strings.ToLower(s)
    s = slugRe.ReplaceAllString(s, "-")
    s = strings.Trim(s, "-")
    if s == "" || s == "." || s == ".." {
        return "_"
    }
    return s
}

func stripFirstH1(content string) string {
    first, rest, found := strings.Cut(content, "\n")
    if strings.HasPrefix(strings.TrimSpace(first), "# ") {
        if found {
            return rest
        }
        return ""
    }
    return content
}

func firstNonEmptyLine(content string) string {
    scanner := bufio.NewScanner(strings.NewReader(content))
    for scanner.Scan() {
        l := strings.TrimSpace(scanner.Text())
        if l == "" || strings.HasPrefix(l, "![") || strings.HasPrefix(l, "<img") {
            continue
        }
        return l
    }
    return ""
}

// rewriteImageURLs заменяет относительные ссылки на изображения в тексте README
// на абсолютные URL вида https://raw.githubusercontent.com/owner/repo/branch/...
// чтобы картинки отображались на Hugo-сайте.
func rewriteImageURLs(content string, p Project) string {
    owner, name, err := parseRepoURL(p.Repo)
    if err != nil {
        return content
    }
    baseRaw := fmt.Sprintf("https://raw.githubusercontent.com/%s/%s/%s",
        owner, name, p.Branch)

    // Markdown: ![alt](url) и ![alt](url "title")
    // Группы: 1 = "![alt](", 2 = url
    mdRe := regexp.MustCompile(`(!\[[^\]]*\]\()([^)\s]+)`)
    content = mdRe.ReplaceAllStringFunc(content, func(m string) string {
        sub := mdRe.FindStringSubmatch(m)
        if len(sub) < 3 || isAbsoluteURL(sub[2]) {
            return m
        }
        rel := strings.TrimPrefix(sub[2], "./")
        return sub[1] + baseRaw + "/" + rel
    })

    // HTML: <img src="url"> и <img src='url'>
    // Группы: 1 = все до url, 2 = кавычка, 3 = url, 4 = закрывающая кавычка
    htmlRe := regexp.MustCompile(`(<img\b[^>]*\bsrc=)(["'])([^"']+)(["'])`)
    content = htmlRe.ReplaceAllStringFunc(content, func(m string) string {
        sub := htmlRe.FindStringSubmatch(m)
        if len(sub) < 5 || isAbsoluteURL(sub[3]) {
            return m
        }
        rel := strings.TrimPrefix(sub[3], "./")
        return sub[1] + sub[2] + baseRaw + "/" + rel + sub[4]
    })

    return content
}

func isAbsoluteURL(s string) bool {
    return strings.HasPrefix(s, "http://") ||
        strings.HasPrefix(s, "https://") ||
        strings.HasPrefix(s, "//") ||
        strings.HasPrefix(s, "#") ||
        strings.HasPrefix(s, "mailto:")
}

func stripFirstImage(content string) string {
    re := regexp.MustCompile(`!\[[^\]]*\]\([^)\s]+\)`)
    done := false
    result := re.ReplaceAllStringFunc(content, func(m string) string {
        if done || strings.Contains(m, "shields.io") {
            return m
        }
        done = true
        return ""
    })
    // убираем строки, ставшие пустыми после удаления изображения
    emptyLineRe := regexp.MustCompile(`\n{3,}`)
    return emptyLineRe.ReplaceAllString(result, "\n\n")
}