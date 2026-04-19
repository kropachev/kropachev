package main

import (
    "fmt"
    "io"
    "net/http"
    "regexp"
    "strings"
)

var ogImageRe = regexp.MustCompile(`<meta\s+property="og:image"\s+content="([^"]+)"`)
var mdImageRe = regexp.MustCompile(`!\[[^\]]*\]\(([^)\s]+)`)

func fetchCoverImage(p Project, readmeContent string) ([]byte, string, error) {
    if data, ext, err := fetchSocialPreview(p); err == nil {
        return data, ext, nil
    }

    imageURL := firstReadmeImageURL(p, readmeContent)
    if imageURL == "" {
        return nil, "", fmt.Errorf("no cover image found")
    }
    return downloadImage(imageURL)
}

func fetchSocialPreview(p Project) ([]byte, string, error) {
    owner, name, err := parseRepoURL(p.Repo)
    if err != nil {
        return nil, "", err
    }

    pageURL := fmt.Sprintf("https://github.com/%s/%s", owner, name)
    req, err := http.NewRequest(http.MethodGet, pageURL, nil)
    if err != nil {
        return nil, "", err
    }
    req.Header.Set("User-Agent", "fetch-projects/1.0")

    resp, err := httpClient.Do(req)
    if err != nil {
        return nil, "", err
    }
    defer resp.Body.Close()

    if resp.StatusCode != http.StatusOK {
        return nil, "", fmt.Errorf("fetch %s: %s", pageURL, resp.Status)
    }

    // Meta-теги находятся в <head>, достаточно первых 50 КБ
    head, err := io.ReadAll(io.LimitReader(resp.Body, 50*1024))
    if err != nil {
        return nil, "", err
    }

    m := ogImageRe.FindSubmatch(head)
    if len(m) < 2 {
        return nil, "", fmt.Errorf("og:image not found")
    }

    imageURL := string(m[1])

    // Автосгенерированные превью — не настоящие обложки
    if strings.Contains(imageURL, "opengraph.githubassets.com") {
        return nil, "", fmt.Errorf("no custom social preview")
    }

    return downloadImage(imageURL)
}

func firstReadmeImageURL(p Project, content string) string {
    for _, m := range mdImageRe.FindAllStringSubmatch(content, -1) {
        if len(m) < 2 || m[1] == "" {
            continue
        }
        url := m[1]
        if strings.Contains(url, "shields.io") {
            continue
        }
        if !isAbsoluteURL(url) {
            owner, name, err := parseRepoURL(p.Repo)
            if err != nil {
                continue
            }
            url = strings.TrimPrefix(url, "./")
            url = fmt.Sprintf("https://raw.githubusercontent.com/%s/%s/%s/%s",
                owner, name, p.Branch, url)
        }
        return url
    }
    return ""
}

func downloadImage(imageURL string) ([]byte, string, error) {
    resp, err := httpClient.Get(imageURL)
    if err != nil {
        return nil, "", err
    }
    defer resp.Body.Close()

    if resp.StatusCode != 200 {
        return nil, "", fmt.Errorf("fetch image %s: %s", imageURL, resp.Status)
    }

    data, err := io.ReadAll(io.LimitReader(resp.Body, 10*1024*1024))
    if err != nil {
        return nil, "", err
    }

    ext := extFromContentType(resp.Header.Get("Content-Type"))
    return data, ext, nil
}

func extFromContentType(ct string) string {
    switch {
    case strings.Contains(ct, "image/png"):
        return ".png"
    case strings.Contains(ct, "image/jpeg"):
        return ".jpg"
    case strings.Contains(ct, "image/gif"):
        return ".gif"
    case strings.Contains(ct, "image/webp"):
        return ".webp"
    default:
        return ".png"
    }
}