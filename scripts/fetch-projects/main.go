package main

import (
    "fmt"
    "net/http"
    "os"
    "path/filepath"
    "sync"
    "time"

    "gopkg.in/yaml.v3"
)

var httpClient = &http.Client{Timeout: 30 * time.Second}

type Project struct {
    Repo        string `yaml:"repo"`
    Slug        string `yaml:"slug"`
    Branch      string `yaml:"branch"`
    Title       string `yaml:"title"`
    Description string `yaml:"description"`
    FetchImage  *bool  `yaml:"fetch_image"`
    StripCover  bool   `yaml:"strip_cover"`
}

func main() {
    projects, err := loadProjects("data/projects.yaml")
    if err != nil {
        panic(err)
    }

    for i := range projects {
        if projects[i].Slug == "" {
            _, name, err := parseRepoURL(projects[i].Repo)
            if err == nil {
                projects[i].Slug = sanitizeSlug(name)
            }
        }
        if projects[i].Branch == "" {
            projects[i].Branch = "main"
        }
    }

    if err := cleanupProjects(projects); err != nil {
        fmt.Fprintf(os.Stderr, "cleanup: %v\n", err)
    }

    const maxConcurrency = 5
    sem := make(chan struct{}, maxConcurrency)
    var wg sync.WaitGroup

    for _, p := range projects {
        wg.Add(1)
        go func(p Project) {
            defer wg.Done()
            sem <- struct{}{}
            defer func() { <-sem }()

            if err := syncProject(p); err != nil {
                fmt.Fprintf(os.Stderr, "project %s: %v\n", p.Slug, err)
            }
        }(p)
    }
    wg.Wait()
}

func loadProjects(path string) ([]Project, error) {
    f, err := os.Open(path)
    if err != nil {
        return nil, err
    }
    defer f.Close()

    var projects []Project
    dec := yaml.NewDecoder(f)
    if err := dec.Decode(&projects); err != nil {
        return nil, err
    }
    return projects, nil
}

func cleanupProjects(projects []Project) error {
    dir := filepath.Join("content", "projects")
    entries, err := os.ReadDir(dir)
    if err != nil {
        if os.IsNotExist(err) {
            return nil
        }
        return err
    }

    expected := make(map[string]bool)
    for _, p := range projects {
        expected[p.Slug] = true
    }

    for _, e := range entries {
        if !e.IsDir() {
            continue
        }
        if !expected[e.Name()] {
            path := filepath.Join(dir, e.Name())
            fmt.Println("Removing", path)
            if err := os.RemoveAll(path); err != nil {
                return err
            }
        }
    }
    return nil
}