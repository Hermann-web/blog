---
date: 2024-05-04
authors: [hermann-web]
comments: true
description: >
  Explore numerical analysis and plotting syntax equivalence across Python, R, MATLAB, and Scilab for effective cross-platform development.
categories:
  - programming
  - numerical-analysis
  - data-visualization
  - tools-comparison
links:
  - blog/posts/a-guide-to-numerical-analysis.md
  - blog/posts/data-visualization-with-seaborn.md
title: "Numerical Analysis and Plotting: Equivalence in Python, R, MATLAB, and Scilab"
---

# Numerical Analysis and Plotting: Equivalence in Python, R, MATLAB, and Scilab

## Introduction

Numerical analysis and data visualization are fundamental aspects of scientific computing across various programming languages.

<div class="float-img-container float-img-right">
  <a title="Bill Casselman, CC BY-SA 3.0 &lt;http://creativecommons.org/licenses/by-sa/3.0/&gt;, via Wikimedia Commons" href="https://commons.wikimedia.org/wiki/File:Ybc7289-bw.jpg"><img width="256" alt="Ybc7289-bw" src="https://upload.wikimedia.org/wikipedia/commons/0/0b/Ybc7289-bw.jpg"></a>
</div>

This guide explores syntax equivalences in Python, R, MATLAB, and Scilab, empowering you to transition seamlessly between these languages for numerical computations and plotting tasks.

Understanding the corresponding syntaxes in each language facilitates code portability and collaboration among researchers and developers across different platforms.

<!-- more -->

## Key Considerations

### Choosing the Right Language

- **Syntax Familiarity:** Consider your familiarity with the language syntax.
- **Library Ecosystem:** Evaluate the availability and maturity of numerical and plotting libraries.
- **Community Support:** Assess the community size and active development in relevant domains.
- **Integration Requirements:** Determine integration needs with existing systems or frameworks.
- **Performance:** Consider the performance requirements for your numerical computations.

### Languages Overview

=== ":octicons-file-code-16: Python"

    - **Numerical Analysis Libraries:** NumPy, SciPy
    - **Plotting Libraries:** Matplotlib, sns
    - **Syntax Highlights:** Vectorized operations, array slicing, and broadcasting.

=== ":octicons-file-code-16: R"

    - **Numerical Analysis Libraries:** Base R functions, `data.table`, `dplyr`
    - **Plotting Libraries:** ggplot2, plotly
    - **Syntax Highlights:** Data frames, functional programming, pipe operator `%>%`.

=== ":octicons-file-code-16: MATLAB"

    - **Numerical Analysis Libraries:** MATLAB Core Functions, MATLAB Toolboxes
    - **Plotting Libraries:** MATLAB Plotting Functions
    - **Syntax Highlights:** Matrix operations, built-in functions, plotting with `plot` and `imshow`.

=== ":octicons-file-code-16: Scilab"

    - **Numerical Analysis Libraries:** Scilab Core Functions, Toolboxes
    - **Plotting Libraries:** Scilab Plotting Functions
    - **Syntax Highlights:** Matrix-based computations, built-in functions, plotting with `plot` and `imshow`.

## Comparison Tables

### Array Creation

| Task                          | Python (NumPy)                  | R (Base/Additional Packages)             | MATLAB Core Functions     | Scilab Core Functions     |
| ----------------------------- | ------------------------------- | ---------------------------------------- | ------------------------- | ------------------------- |
| Row Vector                    | `np.array([1, 2, 3])`           | `c(1, 2, 3)`                             | `row_vector = [1, 2, 3]`  | `row_vector = [1, 2, 3]`  |
| Column Vector                 | `np.array([[1], [2], [3]])`     | `c(1, 2, 3)`                             | `col_vector = [1; 2; 3]`  | `col_vector = [1; 2; 3]`  |
| Matrix                        | `np.array([[1, 2], [3, 4]])`    | `matrix(c(1, 2, 3, 4), nrow=2)`          | `matrix = [1, 2; 3, 4]`   | `matrix = [1, 2; 3, 4]`   |
| j:k Row Vector                | `np.arange(j, k+1)`             | `j:k`                                    | `j:k`                     | `j:k`                     |
| j:i:k Row Vector              | `np.arange(j, k+i, i)`          | `seq(j, k, by=i)`                        | `j:i:k`                   | `j:i:k`                   |
| Linearly Spaced Vector        | `np.linspace(a, b, n)`          | `seq(a, b, length.out=n)`                | `linspace(a, b, n)`       | `linspace(a, b, n)`       |
| Linearly spaced vector        | `np.linspace(1, 10, 100)`       | `seq(1, 10, length.out=100)`             | `linspace(1, 10, 100)`    | `linspace(1, 10, 100)`    |
| Logarithmically spaced vector | `np.logspace(1, 3, 100)`        | `10^seq(1, 3, length.out=100)`           | `logspace(1, 3, 100)`     | `logspace(1, 3, 100)`     |
| Range Array                   | `np.arange(start, stop, step)`  | `seq(from = start, to = end, by = step)` | `start:step:end`          | `start:step:end`          |
| Empty Array                   | `np.empty(shape=(m, n))`        | `vector(length = n)`                     | `zeros(m, n)`             | `zeros(m, n)`             |
| NaN Matrix                    | `np.full((a, b), np.nan)`       | `matrix(NaN, nrow=a, ncol=b)`            | `NaN(a, b)`               | `nan(a, b)`               |
| Zeros Array                   | `np.zeros(shape=(m, n))`        | `rep(0, n)`                              | `zeros(m, n)`             | `zeros(m, n)`             |
| Zeros Matrix                  | `np.zeros((a, b))`              | `matrix(0, nrow=a, ncol=b)`              | `zeros(a, b)`             | `zeros(a, b)`             |
| Ones Array                    | `np.ones(shape=(m, n))`         | `rep(1, n)`                              | `ones(m, n)`              | `ones(m, n)`              |
| Ones Matrix                   | `np.ones((a, b))`               | `matrix(1, nrow=a, ncol=b)`              | `ones(a, b)`              | `ones(a, b)`              |
| Identity Matrix               | `np.identity(n)` or `np.eye(n)` | `diag(rep(1, n))`                        | `eye(n)`                  | `eye(n)`                  |
| Meshgrid                      | `np.meshgrid(x, y)`             | `expand.grid(x, y)`                      | `[X, Y] = meshgrid(x, y)` | `[X, Y] = meshgrid(x, y)` |
| Random Array                  | `np.random.rand(rows, cols)`    | `runif(n)`                               | `rand(m, n)`              | `rand(m, n)`              |

### Descriptive Methods

| Task                          | Python                     | R      | MATLAB Core Functions             | Scilab Core Functions             |
|-------------------------------|-------------------------------------|-----------------------------------|----------------------------------|----------------------------------|
| Rows and Columns      | `x.shape`                               | `dim(x)`                                | `size(x)`                               | `size(x)`                               |
| Number of Array Elements | `np.size(A)`                        | `length(A)`                             | `numel(A)`                              | `length(A)`                             |

### Accessing Elements

| Feature               | Python (NumPy/SciPy)                   | R (Base/Additional Packages)            | MATLAB Core Functions                   | Scilab Core Functions                   |
|-----------------------|-----------------------------------------|-----------------------------------------|-----------------------------------------|-----------------------------------------|
| Change Index Value    | `x[1] = 4`                              | `x[2] <- 4`                             | `x(2) = 4`                              | `x(2) = 4`                              |
| All Elements of x     | `x.flatten()`                           | `as.vector(x)`                          | `x(:)`                                  | `x(:)`                                  |
| Jth to Last Element   | `x[j-1:]`                               | `x[j:length(x)]`                        | `x(j:end)`                              | `x(j:$)`                                |
| 2nd to 5th Element    | `x[1:4]`                                | `x[2:5]`                                | `x(2:5)`                                | `x(2:5)`                                |
| All J Row Elements    | `x[j-1, :]`                             | `x[j, ]`                                | `x(j, :)`                               | `x(j, :)`                               |
| All J Column Elements | `x[:, j-1]`                             | `x[, j]`                                | `x(:, j)`                               | `x(:, j)`                               |
| Sort Vector           | `np.sort(A)`                            | `sort(A)`                               | `sort(A)`                               | `sort(A)`                               |
| Change Elements > 5 to 0 | `x[x > 5] = 0`                         | `x[x > 5] <- 0`                         | `x(x > 5) = 0`                          | `x(x > 5) = 0`                          |
| List Elements > 5    | `x[x > 5]`                              | `x[x > 5]`                              | `x(x > 5)`                              | `x(x > 5)`                              |
| Indices of Elements > 5 | `np.where(x > 5)`                      | `which(x > 5)`                          | `find(x > 5)`                           | `find(x > 5)`                           |
| Indices of NaN Elements | `np.where(np.isnan(A))`                 | `which(is.na(A))`                       | `find(isnan(A))`                        | `find(isnan(A))`                        |

### Mathematical Operations

| Task                          | Python (NumPy)                      | R (Base/Additional Packages)      | MATLAB Core Functions             | Scilab Core Functions             |
|-------------------------------|-------------------------------------|-----------------------------------|----------------------------------|----------------------------------|
| Element-wise Multiply | `np.multiply(x, y)` or `a * b`                     | `x * y`                                 | `x .* y`                                | `x .* y`                                |
| Element-wise Divide   | `np.divide(x, y)` or `a / b`                       | `x / y`                                 | `x ./ y`                                | `x ./ y`                                |
| Element-wise Add      | `np.add(x, y)` or `a + b`                          | `x + y`                                 | `x + y`                                 | `x + y`                                 |
| Element-wise Subtract | `np.subtract(x, y)` or `a - b`                     | `x - y`                                 | `x - y`                                 | `x - y`                                 |
| Element-wise square           | `np.square(a)`                      | `a^2`                             | `a.^2`                           | `a.^2`                           |
| Elementwise Power     | `np.power(A, n)`                        | `A^n`                                   | `A .^ n`                                | `A .^ n`                                |
| Normal/Matrix Power   | `np.linalg.matrix_power(A, n)`          | `%*%`                                   | `A^n`                                   | `A^n`                                   |
| Diagonal of a matrix | `np.diag(a)`                     | `diag(a)`                           | `diag(a)`                           | `diag(a)`                           |
| Transpose             | `np.transpose(A)`                       | `t(A)`                                  | `A'`                                    | `A'`                                    |
| Horizontally Concatenates | `np.concatenate((A, B), axis=1)`     | `cbind(A, B)`                           | `[A, B]`                                | `[A, B]`                                |
| Vertically Concatenates | `np.concatenate((A, B), axis=0)`       | `rbind(A, B)`                           | `[A; B]`                                | `[A; B]`                                |

### Other Numrical Analysis Related Programming tools

| Task                          | Python                     | R      | MATLAB Core Functions             | Scilab Core Functions             |
|-------------------------------|-------------------------------------|-----------------------------------|----------------------------------|----------------------------------|
| Variable Declaration           | `[a, b] = deal(np.full((5, 5), np.nan))`     | `list(a = matrix(NaN, nrow=5, ncol=5), b = matrix(NaN, nrow=5, ncol=5))` | `[a, b] = deal(NaN(5, 5))`              | `[a, b] = deal(NaN(5, 5))`              |

### Standard Functions

| Function       | Python (Base/NumPy)        | R (Base/Additional Packages) | MATLAB Core Functions | Scilab Core Functions |
|----------------|-----------------------|-------------------------------|-----------------------|-----------------------|
| Absolute Value | `abs(x)` or `np.abs(x)`           | `abs(x)`                      | `abs(x)`              | `abs(x)`              |
| Pi             | `math.pi` or `np.pi`               | `pi`                          | `pi`                  | `%pi`                 |
| Infinity       | `math.inf` or `np.inf`              | `Inf`                         | `Inf`                 | `%inf`                |
| Floating Point Accuracy | `np.finfo(float).eps` | `double.eps`              | `eps`                 | `eps`                 |
| Large Number   | `1e6`                 | `1e6`                         | `1e6`                 | `1d6`                 |
| Sum            | `np.sum(x)`           | `sum(x)`                      | `sum(x)`              | `sum(x)`              |
| Cumulative Sum | `np.cumsum(x)`        | `cumsum(x)`                   | `cumsum(x)`           | `cumsum(x)`           |
| Product        | `np.prod(x)`          | `prod(x)`                     | `prod(x)`             | `prod(x)`             |
| Cumulative Product | `np.cumprod(x)`    | `cumprod(x)`                  | `cumprod(x)`          | `cumprod(x)`          |
| Difference     | `np.diff(x)`          | `diff(x)`                     | `diff(x)`             | `diff(x)`             |
| Round          | `np.round(x)`         | `round(x)`                    | `round(x)`            | `round(x)`            |
| Ceiling        | `np.ceil(x)`          | `ceiling(x)`                  | `ceil(x)`             | `ceil(x)`             |
| Floor          | `np.floor(x)`         | `floor(x)`                    | `floor(x)`            | `floor(x)`            |
| Bessel Function| `scipy.special.jn(n, x)` | `besselJ(x, n)`           | `besselj(n, x)`       | `besselj(n, x)`       |
| Factorial      | `np.math.factorial(x)`| `factorial(x)`                | `factorial(x)`        | `factorial(x)`        |
| Exponential function          | `np.exp(a)`                      | `exp(a)`                          | `exp(a)`                         | `exp(a)`                         |
| Square root                   | `np.sqrt(a)`                     | `sqrt(a)`                         | `sqrt(a)`                        | `sqrt(a)`                        |
| Trigonometric functions       | `np.sin(a)`, `np.cos(a)`, `np.tan(a)` | `sin(a)`, `cos(a)`, `tan(a)` | `sin(a)`, `cos(a)`, `tan(a)` | `sin(a)`, `cos(a)`, `tan(a)` |

### Statistical Measures

| Measure               | Python (NumPy/SciPy)                               | R (Base/Additional Packages)               | MATLAB Core Functions                    | Scilab Core Functions                   |
|-----------------------|----------------------------------------------------|--------------------------------------------|------------------------------------------|-----------------------------------------|
| Mean                  | `np.mean(x)` or `np.average(x)`                     | `mean(x)`                                  | `mean(x)` or `mean(mean(x))`             | `mean(x)` or `mean(x(:))`               |
| Median                | `np.median(x)`                                     | `median(x)`                                | `median(x)`                              | `median(x)`                            |
| Variance              | `np.var(x)`                                        | `var(x)`                                   | `var(x)`                                 | `variance(x)`                         |
| Covariance            | `np.cov(x, y)`                                     | `cov(x, y)`                                | `cov(x, y)`                              | `covar(x, y)`                         |
| Correlation           | `np.corrcoef(x, y)`                                | `cor(x, y)`                                | `corrcoef(x, y)`                         | `correlation(x, y)`                   |
| Quantile              | `np.percentile(x, p)`                             | `quantile(x, p)`                           | `quantile(x, p)`                         | `quantile(x, p)`                       |

*Note: The `quantile(x, p)` function in MATLAB and Scilab uses interpolation for missing quantiles, which may differ from textbook definitions.*

### Statistical Commands

| Command                  | Python (NumPy/SciPy)                             | R (Base/Additional Packages)               | MATLAB Core Functions                    | Scilab Core Functions                   |
|--------------------------|--------------------------------------------------|--------------------------------------------|------------------------------------------|-----------------------------------------|
| Generate Random Numbers  | `np.random.rand(n)` or `np.random.randn(n)`      | `runif(n)` or `rnorm(n)`                   | `rand(n, 1)` or `randn(n, 1)`            | `rand(n, 1)` or `randn(n, 1)`           |
| Probability Density Function | `scipy.stats.norm.pdf(x, mean, std)`         | `dnorm(x, mean, sd)`                       | `normpdf(x, mean, std)`                  | `pdf('normal', x, mean, std)`           |
| Cumulative Distribution Function | `scipy.stats.norm.cdf(x, mean, std)`      | `pnorm(x, mean, sd)`                       | `normcdf(x, mean, std)`                  | `cdf('normal', x, mean, std)`           |
| Histogram                | `plt.hist(x)` or `np.histogram(x)`              | `hist(x)`                                  | `hist(x)`                                | `histogram(x)`                          |
| Histogram with Fit       | `plt.hist(x, density=True)` and `scipy.stats.norm.fit(x)` | `histfit(x)`                           | `histfit(x)`                             | `histfit(x)`                            |

#### Standard Distributions (dist)

- Normal Distribution (norm)
- Student's t-Distribution (t)
- F-Distribution (f)
- Gamma Distribution (gam)
- Chi-Square Distribution (chi2)
- Binomial Distribution (bino)

### Non Linear Numerical Methods

| Feature               | Python (NumPy/SciPy)                   | R (Base/Additional Packages)            | MATLAB Core Functions                   | Scilab Core Functions                   |
|-----------------------|-----------------------------------------|-----------------------------------------|-----------------------------------------|-----------------------------------------|
| Inverse of Matrix     | `np.linalg.inv(A)`                      | `solve(A)`                              | `inv(A)`                                | `inv(A)`                                |
| Eigenvalues & Eigenvectors | `np.linalg.eig(a)`          | `eigen(a)` or (`eigen(A)$values`, `eigen(A)$vectors`)                              | `eig(a)`                                | `spec(a)`                               |
| Singular Value Decomposition | `np.linalg.svd(a)`         | `svd(a)`                                | `svd(a)`                                | `svd(a)`                                |
| Interpolation     | `scipy.interpolate.interp1d(x, y)`    | `approx(x, y)`                          | `interp1(x, y)`                         | `splin(x, y)`                          |
| Quad Integration      | `scipy.integrate.quad(fun, a, b)`      | `integrate.quad(fun, a, b)`            | `quad(fun, a, b)`                       | `quad(fun, a, b)`                       |
| Simpson Integration   | `scipy.integrate.simps(y, x)`           | `integrate.simps(y, x)`                | `simpson(y, x)`                         | `simpson(y, x)`                         |
| Minimization (Derivative-free) | `scipy.optimize.fmin(fun, x0)`  | `optim(x0, fun)`                        | `fminsearch(fun, x0)`                   | `fminsearch(fun, x0)`                   |
| Minimization (Constrained) | `scipy.optimize.minimize(fun, x0, constraints=cons)` | `optim(x0, fun, method = "L-BFGS-B")` | `fmincon(fun, x0, A, b, Aeq, beq, lb, ub)` | `fmincon(fun, x0, A, b, Aeq, beq, lb, ub)` |

### Plotting Methods

| Task                  | Python (Matplotlib/Seaborn)       | R (Base/ggplot2/plotly)               | MATLAB Core Functions               | Scilab Core Functions               |
|-----------------------|------------------------------------|---------------------------------------|-------------------------------------|-------------------------------------|
| Plot line             | `plt.plot(x, y)`                   | `plot(x, y, type='l')` or `ggplot(data, aes(x, y)) + geom_line()`                | `plot(x, y)`                        | `plot(x, y)` or `plot2d(x, y)`                      |
| Scatter plot          | `plt.scatter(x, y)`                | `plot(x, y)` or `ggplot(data, aes(x, y)) + geom_point()`                        | `scatter(x, y)`                     | `plot2d(x, y, style='o')`           |
| Title                 | `plt.title('Title')`               | `title('Title')`                      | `title('Title')`                    | `xtitle('Title')`                   |
| Label axes            | `plt.xlabel('x'), plt.ylabel('y')` | `xlab('x'), ylab('y')`                | `xlabel('x'), ylabel('y')`          | `xlabel('x'), ylabel('y')`          |
| Histogram             | `plt.hist(data)`                   | `ggplot(data, aes(x)) + geom_histogram()` | `hist(data)`                        | `histplot(data)`                    |
| Heatmap               | `sns.heatmap(data)`                | `plot_ly(z = ~matrix_data, type = "heatmap")` | `heatmap(data)`                   | `heatmap(data)`                     |

### Strings and Regular Expressions

| Task                           | Python (NumPy/SciPy)                          | R (Base/Additional Packages)               | MATLAB Core Functions                   | Scilab Core Functions                   |
|--------------------------------|-----------------------------------------------|--------------------------------------------|-----------------------------------------|-----------------------------------------|
| Compare Strings               | `str1 == str2`                               | `str1 == str2`                             | `strcmp(str1, str2)`                    | `strcmp(str1, str2)`                    |
| Compare Strings (Case Insensitive) | `str1.lower() == str2.lower()`            | `tolower(str1) == tolower(str2)`           | `strcmpi(str1, str2)`                   | `strcmpi(str1, str2)`                   |
| Compare First n Letters       | `str1[:n] == str2[:n]`                        | `substr(str1, 1, n) == substr(str2, 1, n)` | `strncmp(str1, str2, n)`                | `strncmp(str1, str2, n)`                |
| Find Substring                | `str1.find(substring)`                        | `grep(substring, str1)`                    | `strfind(str1, substring)`              | `strindex(str1, substring)`            |
| Regular Expression Search     | `re.search(pattern, str1)`                    | `grep(pattern, str1)`                      | `regexp(str1, pattern)`                 | `regexp(str1, pattern)`                 |

### Logical Operators

| Operator              | Python (NumPy/SciPy)           | R (Base/Additional Packages)   | MATLAB Core Functions    | Scilab Core Functions    |
|-----------------------|--------------------------------|---------------------------------|---------------------------|---------------------------|
| Short-Circuit AND     | `a and b` or `np.logical_and(a, b)` | `a & b`                      | `a && b`                  | `a & b`                   |
| Short-Circuit OR      | `a or b` or `np.logical_or(a, b)`  | `a | b`                      | `a || b`                  | `a | b`                   |
| NOT                   | `not a` or `np.logical_not(a)`     | `!a`                         | `~a`                      | `~a`                      |
| Equality Comparison   | `a == b` or `np.equal(a, b)`        | `a == b`                     | `a == b`                  | `a == b`                  |
| Not Equal             | `a != b` or `np.not_equal(a, b)`    | `a != b`                     | `a ~= b`                  | `a ~= b`                  |
| Object in Class       | `isinstance(obj, class_name)`       | `class(obj) == "class_name"` | `isa(obj, 'class_name')` | `typeof(obj) == 'class_name'` |

### Installing and Importing Libraries

| Feature               | Python (NumPy/SciPy)                   | R (Base/Additional Packages)            | MATLAB Core Functions                   | Scilab Core Functions                   |
|-----------------------|-----------------------------------------|-----------------------------------------|-----------------------------------------|-----------------------------------------|
| Install Library       | `!pip install library_name`            | `install.packages("package_name")`      | Download from MathWorks website or use MATLAB's Add-On Explorer | Download from Scilab website or use ATOMS Package Manager |
| Import Library        | `import library_name`                  | `library(package_name)`                  | N/A                                     | `exec('load' + newline + 'load("path_to_library")')` |

### Debugging and Profiling

| Feature                   | Python (NumPy/SciPy)            | R (Base/Additional Packages)       | MATLAB Core Functions             | Scilab Core Functions             |
|---------------------------|----------------------------------|------------------------------------|----------------------------------|----------------------------------|
| Keyboard Pause Execution  | `KeyboardInterrupt`              | `Sys.sleep()`                      | `keyboard`                       | `pause`                          |
| Return Resumes Execution  | N/A                              | `return()`                         | `return`                         | N/A                              |
| Start Timer               | `start = time.time()`            | `start_time <- Sys.time()`         | `tic`                            | `tic()`                          |
| Stop Timer                | `end = time.time()`              | `end_time <- Sys.time()`           | `toc`                            | `toc()`                          |
| Start Profiler            | N/A                              | `Rprof()`                          | `profile on`                     | `profile on`                     |
| View Profiler Output      | N/A                              | `summaryRprof()`                   | `profile viewer`                 | `profile viewer`                 |
| Try/Catch                 | `try: ... except Exception as e:` | `tryCatch({...}, error = function(e) {...})` | `try ... catch ... end`         | `try ... catch ... end`         |
| Debugging Conditional     | N/A                              | `browser()`                        | `dbstop if error`                | `dbstop if error`                |
| Clear Breakpoints         | N/A                              | N/A                                | `dbclear`                        | `dbclear`                        |
| Resume Execution          | `continue`                       | `next`                             | `dbcont`                         | `dbcont`                         |
| Last Error Message        | `traceback.format_exc()`         | `last.error()`                     | `lasterr`                        | N/A                              |
| Last Warning Message      | N/A                              | `last.warning()`                   | `lastwarn`                       | N/A                              |
| Break                     | N/A                              | `break()`                          | `break`                          | `break`                          |
| Progress Indicator        | `tqdm` or `progressbar`          | `txtProgressBar()`                 | `waitbar`                        | `waitbar`                        |

### Data Import/Export

| Feature                | Python (NumPy/Pandas/Pillow)           | R (Base/Additional Packages)      | MATLAB Core Functions             | Scilab Core Functions             |
|------------------------|---------------------------------|-----------------------------------|----------------------------------|----------------------------------|
| Read Excel             | `pandas.read_excel()`           | `readxl::read_excel()`            | `xlsread()`                      | `xls_read()`                     |
| Write Excel            | `pandas.to_excel()`             | `writexl::write_xlsx()`           | `xlswrite()`                     | `xls_write()`                    |
| Read/Write Table       | `pandas.read_csv()` / `to_csv()` | `readr::read_csv()` / `write_csv()` | `readtable()` / `writetable()` | `csvRead()` / `csvWrite()`      |
| Read/Write Text        | `numpy.loadtxt()` / `savetxt()` | `readr::read_lines()` / `write_lines()` | `dlmread()` / `dlmwrite()`    | `mgetl()` / `mputl()`           |
| Load/Save ASCII        | N/A                             | `load()` / `save()`               | N/A                              | `mget` / `mput`                  |
| Load/Save MATLAB       | `scipy.io.loadmat()` / `savemat()` | N/A                             | `load()` / `save()`               | `loadmat()` / `savemat()`        |
| Read/Write Image       | `PIL.Image.open()` / `save()`   | `readbitmap::read.bmp()` / `write.bmp()` | `imread()` / `imwrite()`     | `imread()` / `imwrite()`         |

### DataFrame Handling

| Feature                           | Python (pandas)                             | R (tidyverse)                            | MATLAB Core Functions                   | Scilab Core Functions                   |
|-----------------------------------|---------------------------------------------|------------------------------------------|-----------------------------------------|-----------------------------------------|
| Import CSV File                   | `import pandas as pd`<br>`fuel_data = pd.read_csv("fuel_data.csv")` | `library(readr)`<br>`fuel_data <- read_csv("fuel_data.csv")` | N/A                                     | N/A                                     |
| View Leading Rows                 | `fuel_data.head(10)`                       | `head(fuel_data)`                        | N/A                                     | N/A                                     |
| List Columns                      | `fuel_data.columns`                        | `names(fuel_data)`                       | `fuel_data.Properties.VariableNames`  | `fieldnames(fuel_data)`                |
| View Metadata                     | `fuel_data.info()`                         | `summary(fuel_data)`                     | N/A                                     | N/A                                     |
| Get Variable Class                | `fuel_data['Price'].dtype`                 | `class(fuel_data$Price)`                 | N/A                                     | N/A                                     |
| Add New Variable                  | `fuel_data['NewVariable'] = some_calculation` | `mutate(fuel_data, NewVariable = some_calculation)` | N/A                                     | N/A                                     |
| Convert to Date                   | `fuel_data['PriceUpdatedDate'] = pd.to_datetime(fuel_data['PriceUpdatedDate'],format="%d/%m/%y")` | `fuel_data$PriceUpdatedDate <- as.Date(fuel_data$PriceUpdatedDate,"%d/%m/%y")`  | N/A                                     | N/A                                     |
| Filter Data by Condition          | `fuel_data[fuel_data['Suburb'] == 'Alexandria']` | `filter(fuel_data, Suburb == "Alexandria")` | N/A                                     | N/A                                     |
| Logical AND Condition             | `fuel_data[(fuel_data['Price'] > 10) & (fuel_data['Price'] < 20)]` | `filter(fuel_data, Price > 10 & Price < 20)` | N/A                                     | N/A                                     |
| Logical OR Condition              | `fuel_data[(fuel_data['Price'] < 5) | (fuel_data['Price'] > 15)]` | `filter(fuel_data, Price < 5 | Price > 15)` | N/A                                     | N/A                                     |

This table provides basic operations for handling data frames in Python (using pandas), R (using tidyverse), MATLAB, and Scilab. Make sure to replace `"fuel_data.csv"` with the actual name of your CSV file.

### Programming Commands

| Feature                | Python (Base/Additional Packages)           | R       | MATLAB Core Functions             | Scilab Core Functions             |
|------------------------|---------------------------------|-----------------------------------|----------------------------------|----------------------------------|
| Return                 | `return`                        | `return`                          | `return`                         | `return`                         |
| Exist                  | `os.path.exists()`              | `exists()`                        | `exist()`                        | `exist()`                        |
| GPU Conversion         | `torch.Tensor.to()`             | `gpuR::gpuVector()`               | `gpuArray()`                     | `gpuArray()`                     |
| Function Declaration   | `def myfun(x1, x2): return x1+x2` | `myfun <- function(x1, x2) { return(x1 + x2) }` | `function [y1,...,yN] = myfun(x1,...,xM)` | `function [y1,...,yN] = myfun(x1,...,xM)` |
| Anonymous Function     | `lambda x1, x2: x1 + x2`       | `function(x1, x2) { x1 + x2 }`   | `myfun = @(x1, x2) x1 + x2`      | `myfun = @(x1, x2) x1 + x2`      |
| Global Scope Declaration       | `global x`                                    | `x <<- value`                             | `global x`                              | `global x`                              |

### System Commands

| Feature             | Python (NumPy/SciPy)    | R (Base/Additional Packages) | MATLAB Core Functions | Scilab Core Functions                           |
| ------------------- | ----------------------- | ---------------------------- | --------------------- | ----------------------------------------------- |
| Add Path            | `sys.path.append(path)` | `attach(path)`               | `addpath(path)`       | `path = path + ":" + path_to_add`               |
| Get Subfolders      | N/A                     | `list.dirs(path)`            | `genpath(path)`       | N/A                                             |
| Current Directory   | `os.getcwd()`           | `getwd()`                    | `pwd`                 | `pwd`                                           |
| Make Directory      | `os.mkdir(path)`        | `dir.create(path)`           | `mkdir(path)`         | `mkdir(path)`                                   |
| Temporary Directory | `tempfile.gettempdir()` | `tempdir()`                  | `tempdir`             | `tempdir()`                                     |
| Functions in Memory | N/A                     | N/A                          | `inmem`               | N/A                                             |
| Exit                | `exit()`                | `q()`                        | `exit`                | `exit`                                          |
| List Folder Content | `os.listdir(path)`      | `list.files(path)`           | `dir(path)`           | `files = dir(path)`                             |
| List Toolboxes      | N/A                     | `installed.packages()`       | `ver`                 | `exec('getmodules' + newline + 'getmodules()')` |

## Conclusion

Congratulations! You've now gained insights into the numerical analysis and plotting syntax equivalence across Python, R, MATLAB, and Scilab. Armed with this knowledge, you can seamlessly switch between these languages for scientific computing tasks, fostering collaboration and code portability.

Stay proactive in exploring and utilizing the rich libraries and functionalities offered by each language to enhance your computational workflows!

## Related Posts

- [Cheat on Python Package Managers](../../posts/programming-languages/python/package-managers-in-python.md)

## Relevant Sources

- [NumPy Documentation](https://np.org/doc/)
- [R Documentation](https://www.rdocumentation.org/)
- [MATLAB Documentation](https://www.mathworks.com/help/matlab/), including [python pandas to matlab - mathworks](https://www.mathworks.com/matlabcentral/fileexchange/111770-pandastomatlab)
- [Scilab Documentation](https://help.scilab.org/), including [struct](https://help.scilab.org/docs/2024.0.0/en_US/struct.html) and [csvread](https://help.scilab.org/docs/2024.0.0/en_US/csvRead.html)
- [Thor Nielsen, Matlab Programming Syntax Cheat](https://sites.nd.edu/gfu/files/2019/07/cheatsheet.pdf)
- [Glen Bentley, R & Python Cheat Sheet](https://www.rpubs.com/Bentley_87/542213)
