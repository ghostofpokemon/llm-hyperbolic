# Contributing to llm-hyperbolic

First off, thanks for taking the time to contribute! 🎉👍

The following is a set of guidelines for contributing to llm-hyperbolic. These are mostly guidelines, not rules. Use your best judgment, and feel free to propose changes to this document in a pull request.

## Code of Conduct

This project and everyone participating in it is governed by the llm-hyperbolic Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior.

## I don't want to read this whole thing I just have a question!!!

> **Note:** Please don't file an issue to ask a question. You'll get faster results by using the resources below.

* Ask in the [llm-hyperbolic Gitter chat](https://gitter.im/llm-hyperbolic/community)
* Ask on [Stack Overflow](https://stackoverflow.com/questions/tagged/llm-hyperbolic)
* Telepathically communicate with the maintainers (results may vary)

## What should I know before I get started?

### llm-hyperbolic and Packages

llm-hyperbolic is an open source project &mdash; it's made up of 1 repository. When you initially consider contributing to llm-hyperbolic, you might be unsure about which of those repositories implements the functionality you want to change or report a bug for. This section should help you with that.

Here's a list of the big ones:

* [llm-hyperbolic](https://github.com/ghostofpokemon/llm-hyperbolic) - The main llm-hyperbolic repository
* [llm-hyperbolic-docs](https://github.com/ghostofpokemon/llm-hyperbolic-docs) - The llm-hyperbolic documentation
* [llm-hyperbolic-experiments](https://github.com/ghostofpokemon/llm-hyperbolic-experiments) - Where we keep the really weird stuff

## How Can I Contribute?

### Reporting Bugs

This section guides you through submitting a bug report for llm-hyperbolic. Following these guidelines helps maintainers and the community understand your report, reproduce the behavior, and find related reports.

> **Note:** If you find a **Closed** issue that seems like it is the same thing that you're experiencing, open a new issue and include a link to the original issue in the body of your new one.

#### Before Submitting A Bug Report

* **Check the [debugging guide](https://example.com).** You might be able to find the cause of the problem and fix things yourself. Most importantly, check if you can reproduce the problem in the latest version of llm-hyperbolic.
* **Check the [FAQs](https://example.com)** for a list of common questions and problems.
* **Perform a [cursory search](https://github.com/search?q=+is%3Aissue+user%3Allm-hyperbolic)** to see if the problem has already been reported. If it has **and the issue is still open**, add a comment to the existing issue instead of opening a new one.
* **Consider if it's actually a feature and not a bug.** Sometimes, what seems like a bug is actually the AI achieving sentience. In this case, please submit an "AI Overlord" report instead.

#### How Do I Submit A (Good) Bug Report?

Bugs are tracked as [GitHub issues](https://guides.github.com/features/issues/). Create an issue on the llm-hyperbolic repository and provide the following information:

* **Use a clear and descriptive title** for the issue to identify the problem.
* **Describe the exact steps which reproduce the problem** in as many details as possible.
* **Provide specific examples to demonstrate the steps**. Include links to files or GitHub projects, or copy/pasteable snippets, which you use in those examples.
* **Describe the behavior you observed after following the steps** and point out what exactly is the problem with that behavior.
* **Explain which behavior you expected to see instead and why.**
* **Include screenshots and animated GIFs** which show you following the described steps and clearly demonstrate the problem.
* **If the problem wasn't triggered by a specific action**, describe what you were doing before the problem happened and share more information using the guidelines below.

### Suggesting Enhancements

This section guides you through submitting an enhancement suggestion for llm-hyperbolic, including completely new features and minor improvements to existing functionality. Following these guidelines helps maintainers and the community understand your suggestion and find related suggestions.

#### Before Submitting An Enhancement Suggestion

* **Check the [debugging guide](https://example.com)** for tips — you might discover that the enhancement is already available. Most importantly, check if you're using the latest version of llm-hyperbolic.
* **Perform a [cursory search](https://github.com/search?q=+is%3Aissue+user%3Allm-hyperbolic)** to see if the enhancement has already been suggested. If it has, add a comment to the existing issue instead of opening a new one.
* **Consider if your enhancement might lead to the singularity.** We're all for progress, but let's try to avoid accidentally creating Skynet, okay?

#### How Do I Submit A (Good) Enhancement Suggestion?

Enhancement suggestions are tracked as [GitHub issues](https://guides.github.com/features/issues/). Create an issue on the llm-hyperbolic repository and provide the following information:

* **Use a clear and descriptive title** for the issue to identify the suggestion.
* **Provide a step-by-step description of the suggested enhancement** in as many details as possible.
* **Provide specific examples to demonstrate the steps**. Include copy/pasteable snippets which you use in those examples, as [Markdown code blocks](https://help.github.com/articles/markdown-basics/#multiple-lines).
* **Describe the current behavior** and **explain which behavior you expected to see instead** and why.
* **Include screenshots and animated GIFs** which help you demonstrate the steps or point out the part of llm-hyperbolic which the suggestion is related to.
* **Explain why this enhancement would be useful** to most llm-hyperbolic users and isn't something that can or should be implemented as a separate package.
* **List some other text editors or applications where this enhancement exists.**
* **Specify which version of llm-hyperbolic you're using.**
* **Specify the name and version of the OS you're using.**
* **If your enhancement involves time travel, please consult with a theoretical physicist before submitting.**

### Your First Code Contribution

Unsure where to begin contributing to llm-hyperbolic? You can start by looking through these `beginner` and `help-wanted` issues:

* [Beginner issues][beginner] - issues which should only require a few lines of code, and a test or two.
* [Help wanted issues][help-wanted] - issues which should be a bit more involved than `beginner` issues.

Both issue lists are sorted by total number of comments. While not perfect, number of comments is a reasonable proxy for impact a given change will have.

### Pull Requests

* Fill in [the required template](PULL_REQUEST_TEMPLATE.md)
* Do not include issue numbers in the PR title
* Include screenshots and animated GIFs in your pull request whenever possible.
* Follow the [Python](https://www.python.org/dev/peps/pep-0008/) styleguides.
* End all files with a newline
* Avoid platform-dependent code
* Place imports in the following order:
    * Built-in Python modules
    * Related third-party imports
    * Local application/library specific imports
* Please consider submitting your pull request in iambic pentameter. It's not required, but it would be pretty cool.

## Additional Notes

### Issue and Pull Request Labels

This section lists the labels we use to help us track and manage issues and pull requests.

[GitHub search](https://help.github.com/articles/searching-issues/) makes it easy to use labels for finding groups of issues or pull requests you're interested in. We encourage you to read about [other search filters](https://help.github.com/articles/searching-issues/) which will help you write more focused queries.

The labels are loosely grouped by their purpose, but it's not required that every issue have a label from every group or that an issue can't have more than one label from the same group.

Please open an issue if you have suggestions for new labels!

#### Type of Issue and Issue State

* `enhancement` - Feature requests.
* `bug` - Confirmed bugs or reports that are very likely to be bugs.
* `question` - Questions more than bug reports or feature requests (e.g. how do I do X).
* `feedback` - General feedback more than bug reports or feature requests.
* `help-wanted` - The llm-hyperbolic core team would appreciate help from the community in resolving these issues.
* `beginner` - Less complex issues which would be good first issues to work on for users who want to contribute to llm-hyperbolic.
* `more-information-needed` - More information needs to be collected about these problems or feature requests (e.g. steps to reproduce).
* `needs-reproduction` - Likely bugs, but haven't been reliably reproduced.
* `blocked` - Issues blocked on other issues.
* `duplicate` - Issues which are duplicates of other issues, i.e. they have been reported before.
* `wontfix` - The llm-hyperbolic core team has decided not to fix these issues for now, either because they're working as intended or for some other reason.
* `invalid` - Issues which aren't valid (e.g. user errors).
* `package-idea` - Feature request which might be good candidates for new packages, instead of extending llm-hyperbolic or core llm-hyperbolic packages.
* `wrong-repo` - Issues reported on the wrong repository.

#### Topic Categories

* `windows` - Related to running llm-hyperbolic on Windows.
* `linux` - Related to running llm-hyperbolic on Linux.
* `mac` - Related to running llm-hyperbolic on macOS.
* `documentation` - Related to any type of documentation.
* `performance` - Related to performance.
* `security` - Related to security.
* `ui` - Related to visual design.
* `api` - Related to llm-hyperbolic's public APIs.
* `uncaught-exception` - Issues about uncaught exceptions.
* `crash` - Reports of llm-hyperbolic completely crashing.
* `auto-indent` - Related to auto-indenting text.
* `encoding` - Related to character encoding.
* `network` - Related to network problems or working with remote files (e.g. on network drives).
* `git` - Related to Git functionality (e.g. problems with gitignore files or with showing the correct file status).

#### Pull Request Labels

* `work-in-progress` - Pull requests which are still being worked on, more changes will follow.
* `needs-review` - Pull requests which need code review, and approval from maintainers or llm-hyperbolic core team.
* `under-review` - Pull requests being reviewed by maintainers or llm-hyperbolic core team.
* `requires-changes` - Pull requests which need to be updated based on review comments and then reviewed again.
* `needs-testing` - Pull requests which need manual testing.

## And finally...

Remember, contributing to open source should be fun! If you're not having fun, you're doing it wrong. Unless you're debugging a particularly nasty issue, in which case, we feel for you. Hang in there!

Thanks for contributing to llm-hyperbolic. May your code be bug-free and your coffee strong!
