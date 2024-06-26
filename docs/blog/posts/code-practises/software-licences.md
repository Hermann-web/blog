---
date: 2023-10-24
authors: [hermann-web]
comments: true
description: |
  Unveiling the Code Chronicles: Navigating the Realm of Software Licenses in Development, Enriched with mkdocs.
categories:
  - Blog
  - beginners
links:
  - blog/posts/a-roadmap-for-web-dev.md
  - blog/posts/code-practises/software-licences.md
title: "Unveiling the Code Chronicles: Navigating Software Licenses"
---

## Introduction

__Ever glanced at those cryptic licenses in software projects—like the `MIT License`, `Apache License`, or `GPL`—and wondered, 'What do they mean for me?'__

Whether you're a code wizard or just dipping your toes in tech, understanding these licenses is like decoding a secret language. Which one suits your project best? What's the deal when you borrow or tweak someone else's code?

This document unravels the mystery, diving into the world of software licenses. Discover their quirks, choose wisely between the MIT, Apache, or GPL licenses, and learn the ropes for handling borrowed or tweaked code. Get ready to crack the code of software licenses!

!!! warning "Please note that the information provided here is for educational purposes only and should not be construed as legal advice."

<!-- more -->

## Motivation for Choosing Licenses

Understanding software licenses is crucial for project development. Choosing a license depends on factors like project goals, desired openness, collaboration, and legal obligations when utilizing or modifying a project.

### What to Choose for Your Project?

- Consider the project's goals, community involvement, and desired freedoms for users when selecting a license.
- Assess the implications of each license on collaboration, distribution, and derivative works.

### Actions with Copied/Modified Projects

- Adhering to the original license terms is crucial when using, modifying, or distributing a project.
- Respect the license obligations to the original authors while exercising your rights under the chosen license.

## Open Source Licenses

A software is `Open source` when his code source is available (Not quite the same for llm though).
It can be permissive (MIT, Apache, ...), copyleft (you should distribute under the same license and/or keep track of the modification and/or not remove the claim in the license).

### Permissive Licenses

They allow anyone to use, modify, distribute, and sell the software with few restrictions. Though you're not obligated to share modifications, distributing in source code form implies sharing. Here are the most common exemples.

=== ":octicons-file-code-16: `MIT License:`"

    - Allows users `U1` to use, modify, and distribute (release to user `U2`) the software ( which a part is a derivative from `U0` work) for any purpose, including commercial use, without having to share ( to user `U2`) their modifications or contributions
    - Essentially, it offers considerable freedom.

=== ":octicons-file-code-16: `Apache License / Microsoft Public License (Ms-PL)`"

    - Permits users `U1` to use, modify, and distribute the software without having to share their modifications with `U2`
    - Requires users to provide attribution (a mention of the original author' name `U0`) and a copy of the license when distributing the software to `U2`.
    - Includes a patent license, ensuring users won't be sued for patents related to the software.
    - So basically, keep the original author's name and no patent (warranties) problem

=== ":octicons-file-code-16: `BSD License`"

    - Allows redistribution and modification
    - Requires that the original copyright notice and disclaimer be retained.
    - Essentially, it ensures the original author (`U0` or `U1`) are not legally responsible for any issues arising from User `U2`'s use.

### Copyleft Licenses

They

- Impose restrictions on how the software can be used and distributed.
- Require that any modifications or derivative works (In programming for example, the part of a software that use/modify a copyleft licenced code) of the software also be released under the same license, maintaining the software's open-source nature.

This helps ensure that the software remains open source and that any improvements made to it are shared with the community. Examples of copyleft licenses include the GNU General Public License (GPL) and the Lesser General Public License (LGPL) or AGPL or  SSPL or BSD (Berkeley Software Distribution) or MPL ( Mozillah Public Licence) or Eclipse Public License (EPL) or CC ( Creative Common)

Examples include

=== ":octicons-file-code-16: `LGPL (Lesser General Public License)`"

    - Requires users (`U1`) to share modifications and contributions (only the part of the software that use a LGPL licenced code) to the software under the same license (LGPL).
    - Mandates `U1` making the source code available to anyone (`U2`) receiving the software and providing a copy of the license along with it.
    - So basically, requires that modifications to the code be released under the same license.

=== ":octicons-file-code-16: `GPL (General Public License)`"

    - Extend LGPL rules
    - Require the entire codebase become GPL licenced if any part uses GPL-licensed code, ensuring the software remains open source.
    - Require the entire codebase should to be shared with user `U2` when distributing the software
    - So basically, requires that any software that use a GPL licenced code become a GPL licence software that share its codebase.

=== ":octicons-file-code-16: `AGPL (Affero General Public License)`"

    - Extends GPL's rules to apply even when software is accessed via a network.

=== ":octicons-file-code-16: `SSPL (Server Side Public License)`"

    - Extend AGPL rules
    - Requires sharing the entire codebase (at least, containing one SSPL-licensed code) with the public (not only user `U2` whose you distribute it to) when using SSPL-licensed code.
    - Mongo DB uses this license.

=== ":octicons-file-code-16: `MPL (Mozilla Public License) / EPL (Eclipse Public License)`"

    - Similar to LGPL, these licenses require modified code portions to be released under the same license but not the entire program. So it allows linking with non-free software
    - Similar to LGPL, requires that modifications to the code be released under the same license

=== ":octicons-file-code-16: `Others`"

    - Microsoft Reciprocal License (Ms-RL)
    - LGPL + disclaimer

## Proprietary Licenses

Those are the most restrictive, controlling software use and distribution and may involve fees. Examples include Microsoft Windows EULA, Adobe Photoshop License Agreement, etc.

## Dual Licensing

Dual licenses use multiple licenses for different parts of software. Challenges arise when incorporating GPL-licensed components as they may necessitate the entire product to be GPL-licensed and shared accordingly with `U2` (those who use your product in a compiled form), unless a separation like mere-aggregation is possible. `Examples like WordPress, which is GPL-licensed, but premium themes might use different licenses.
`

## GPL license: Challenges and Tricks

As a GPL licence requires that any software that use a GPL licenced code become a GPL licence software that share its codebase, these strict obligations can potentially affect the licensing of the entire project.

But if you use a non-free program in your codebase or just don't want to share your codebase, a GPL part is a bug. That's why alternatives like MLP or EPL or LGPL are used to "link" a GPL.

Another trick is to claim a "mere aggregation."  of a (derivative of a gpl licenced component) and (anothers components).
    - Like Android OS which use a modified linux kernel but as the kernel runs at the top of a android os, only the modified linux kernel become gpl. So they have a dual ( Apache licenced Android-OS + Gpl licenced modified-linux-kernel )
    - Or the difference between worpress ( under gpl) vs the themes ( under gpl) (because they use or modify worpress core) vs anothers that don't use or modify the core

## Dual Licensing and Open Source Preferences

This section delves deeper into the dynamics of dual licensing, open source preferences, and their implications within the software development sphere:

### Dual Licensing Consideration

Some view dual licensing as problematic. Users often lean towards open source due to its security, global expertise utilization, and customization capabilities. However, integrating open-source solutions can pose challenges.

### Open Source and Security

Open-source solutions are favored for security reasons and customization but might involve integration challenges. Notable examples include:
    - OpenStack: Uses a permissive Apache license.
    - Mozilla Firefox: Utilizes a permissive MPL (Mozilla Public License).
    - Linux: Governed by GPL (General Public License), chosen for rapid development and innovation adoption.

### Proprietary vs. Open Source

While some prefer proprietary solutions for security and maintenance contracts, others argue in favor of copyleft (e.g., LGPL or GPL). They believe copyleft licenses compel companies to share their work, favoring open source in competition.

### Software Composition Analysis (SCA) Tools

SCA tools automatically scan projects for license compliance, ensuring adherence to licensing requirements.

### Open Source Policy

Establishing an open-source policy involves:
    - Tracking external code licenses and their respective requirements.
    - Utilizing SCA tools for scanning purposes.
    - Maintaining records of license purchases, expiration dates, and repository registrations.
    - Noting that expired licenses may place code in the public domain (if unaltered), while patents and trademarks remain.
    - Keeping up to date with licence change

### License Management

Constant vigilance is required to stay updated with license changes and adhere to compliance standards.

## Conclusion: The Significance of Open Source and Challenges with Dual Licensing

Open source licenses come in various forms, ranging from permissive ones like MIT and Apache to copyleft licenses where you should distribute under the same license and/or keep track of the modification abd/or not remove the claim in the license.  For example, for GPL, if you distribute ( give a software which one part of the code source is under gpl), you should give the source too. It's crucial to understand their implications for your project and how they impact collaboration, distribution, and legal obligations.

Beside open source and proprietary licenses, the dual licences has emerged, combining multiples licenses for differents parts of the same project.

So, dual licensing is a strategy used to address different needs within a project, but it can introduce complexities and limitations. While open source fosters collaboration, innovation, and community-driven development, navigating through licensing choices requires careful consideration and understanding.

Please remember that the information provided in this document is for educational purposes only and should not be considered legal advice. It's essential to consult with legal professionals for specific legal guidance related to software licenses and intellectual property matters.

## Related links

- <https://choosealicense.com>
- <https://www.tldrlegal.com>
