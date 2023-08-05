:html_theme.sidebar_secondary.remove:
.. ablator documentation master file, created by
   sphinx-quickstart on Tue May  2 20:42:43 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
.. only:: not html

   ===================================
   Welcome to ablator!
   ===================================

   .. sidebar:: Right Sidebar Title


.. only:: html

   .. raw:: html
         <html lang="en">
            <head>
               <meta charset="utf-8">
               <title>Documentation.</title>
               
               <link rel="stylesheet" href="./_static/css/index.css">
               <link rel="preconnect" href="https://fonts.googleapis.com">
               <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
               <link
                  href="https://fonts.googleapis.com/css2?family=Dancing+Script:wght@400;500;600;700&family=Fira+Code&family=Roboto:wght@300;400;500;700&display=swap"
                  rel="stylesheet">
               <script>
                  function copyToClipboard(text) {
                        const tempInput = document.createElement('textarea');
                        tempInput.value = text;
                        document.body.appendChild(tempInput);
                        tempInput.select();
                        document.execCommand('copy');
                        document.body.removeChild(tempInput);
                  }
                  function navToPath(path, isNewTab = false) {
                     const newHref = window.location.href.split('#')[0] + path;
                     if (isNewTab) {
                        window.open(newHref, '_blank');
                     } else {
                        window.location.href = newHref;
                     }
                  }        
               </script>
            </head>

            <body>
               <div class="page-root">
                  <div class="page-contents">
                        <div class="banner">
                           <img class="banner-image" src="./_static/ablator-banner-light.svg" alt="ablator-logo">
                           <div class="banner-texts">
                              <h2>Welcome to Ablator Documentations!</h2>
                              <p>Ablator is a robust horizontal-scaling platform for machine learning experiments. You can easily
                                    create parallel-running
                                    experiments with less efforts and less errors. Ablator will take care of the rest.</p>

                              <div class="banner-btn-group">
                                    <button class="custom-btn banner-btn" onclick="window.location.href = `#getting-started`;">
                                       Getting Started
                                    </button>

                                    <div class="banner-codes">
                                       $ pip install ablator

                                       <div class="banner-codes-icon" onclick="copyToClipboard(`pip install ablator`)">
                                          <img height="100%" width="100%" src="./_static/img/copy-icon.png" alt="copy">
                                       </div>
                                    </div>
                                    <a href="https://github.com/fostiropoulos/ablator" target="_blank">
                                       <img class="banner-icon" src="./_static/img/github-mark.png" alt="github">
                                    </a>
                              </div>

                           </div>

                        </div>
                        <div class="contents">
                           <div class="contents-texts">
                              <h3>
                                    Quick Overview
                              </h3>
                              <p>
                                    Here is a quick overview of Ablator documentations' contents. Usages of Ablator are arranged as
                                    following
                                    sections. Please refer to each section for detailed instructions.
                              </p>
                           </div>

                           <div class="contents-grid">

                              <div class="contents-card" onclick="navToPath(`tutorials`)">
                                    <div class="card-title">
                                       <h5>
                                          Basic Tutorials
                                       </h5>
                                    </div>

                                    <p>
                                       The fundamental tutorials of Ablator. Basic usages and contents of Ablator will be
                                       introduced
                                       and explained. Each section will contain a simple demo to elaborate the usage.
                                    </p>
                              </div>
                              <div class="contents-card" onclick="navToPath(`/notebooks/Searchspace-for-diff-optimizers.ipynb`);">
                                    <div class="card-title">
                                       <h5>
                                          Intermediate Tutorials
                                       </h5>
                                    </div>

                                    <p>
                                       Assuming that you have already accumulated some experience with Ablator, this section will
                                       introduce some intermediate usages and contents of Ablator.
                                    </p>
                              </div>
                              <div class="contents-card" onclick="navToPath(`/ablator.html`);">
                                    <div class="card-title">
                                       <h5>
                                          Ablator Packages
                                       </h5>
                                    </div>

                                    <p>
                                       Ablator is composed of several core components packages. Please refer to this section for
                                       detailed usages of
                                       each
                                       component of Ablator.
                                    </p>
                              </div>
                              <div class="contents-card" onclick="navToPath(`/notebooks/GettingStarted-mode-demos.ipynb`);">
                                    <div class="card-title">
                                       <h5>
                                          More Examples
                                       </h5>
                                    </div>
                                    <p>
                                       Ablator is capable of handling various types of deep learning experiments. Please visit this
                                       section for more examples of Ablator use cases.
                                    </p>
                              </div>
                           </div>
                        </div>

                        <div class="basics" id="getting-started">

                           <h3>
                              Getting Started
                           </h3>
                           <div class="features-grid">
                              <div class="feature-card" onclick="navToPath(`/notesbooks/Environment-settings.ipynb`)">
                                    <div class="card-title">
                                       <h5>
                                          Installations
                                       </h5>
                                    </div>
                                    <div class="feature-codes">
                                       $ pip install ablator
                                    </div>

                                    <div class="card-texts">
                                       <p>
                                          Other installation options are also available.
                                       </p>

                                    </div>

                              </div>
                              <div class="feature-card" onclick="navToPath(`/notebooks/GettingStarted.ipynb`)">
                                    <div class="card-title feature-card-title">
                                       <h5>
                                          Quick Start
                                       </h5>
                                    </div>
                                    <div class="card-texts">
                                       <p>
                                          To get started with Ablator quickly, try it out in the demo codes below, where a simple
                                          CNN will be
                                          trained and evaluated with Ablator.
                                       </p>
                                    </div>
                              </div>
                              <div class="feature-card" onclick="navToPath(`/tutorials.html`)">
                                    <div class="card-title feature-card-title">
                                       <h5>
                                          Learn Basics
                                       </h5>
                                    </div>
                                    <div class="card-texts">
                                       <p>
                                          For more basic usages of Ablator, please refer to the Basic Tutorials section below.
                                       </p>
                                    </div>
                              </div>
                           </div>
                        </div>

                        <div class="packages">
                           <div class="contents-texts">
                              <h3>
                                    How Ablator Works
                              </h3>
                              <p>
                                    Ablator is composed of several core components packages. Please refer to this section for
                                    detailed usages of each component of Ablator and learn how Ablator works.
                              </p>
                           </div>



                           <div class="features-grid">
                              <div class="feature-card package-card" onclick="navToPath(`/ablator.config.html`)">
                                    <div class="card-title">
                                       <h5>
                                          Config Package
                                       </h5>
                                    </div>

                                    <div class="card-texts">
                                       <p>
                                          Config Package is where Ablator reads and implements the experiment configurations.
                                       </p>
                                    </div>
                              </div>
                              <div class="feature-card package-card" onclick="navToPath(`/ablator.main.html`)">
                                    <div class="card-title">
                                       <h5>
                                          Main Package
                                       </h5>
                                    </div>

                                    <div class="card-texts">
                                       <p>
                                          Main Package is the core component of Ablator, where most of key functionalities are
                                          integrated.
                                       </p>
                                    </div>
                              </div>

                              <div class="feature-card package-card" onclick="navToPath(`/ablator.modules.html`)">
                                    <div class="card-title">
                                       <h5>
                                          Modules Package
                                       </h5>
                                    </div>

                                    <div class="card-texts">
                                       <p>
                                          Modules Package accommodates the specific modules for execution of the experiments.
                                       </p>
                                    </div>
                              </div>
                              <div class="feature-card package-card" onclick="navToPath(`/ablator.analysis.html`)">
                                    <div class="card-title">
                                       <h5>
                                          Analysis Package
                                       </h5>
                                    </div>

                                    <div class="card-texts">
                                       <p>
                                          Analysis Package is where Ablator analyzes the experiment results.
                                       </p>
                                    </div>
                              </div>
                              <div class="feature-card package-card" onclick="navToPath(`/ablator.utils.html`)">
                                    <div class="card-title">
                                       <h5>
                                          Utils Package
                                       </h5>
                                    </div>

                                    <div class="card-texts">
                                       <p>
                                          In Utils Package, Ablator provides various utility functions for deep learning
                                          experiments.
                                       </p>
                                    </div>
                              </div>
                              <div class="feature-card package-card"
                                    onclick="navToPath(`https://github.com/fostiropoulos/ablator`)">
                                    <div class="card-title">
                                       <h5>
                                          More to come...
                                       </h5>
                                    </div>

                                    <div class="card-texts">
                                       <p>
                                          Ablator is under active development. More features and packages will be added soon...
                                       </p>
                                    </div>
                              </div>
                           </div>
                        </div>

                        <div class="community">
                           <div class="contents-texts">
                              <h3>
                                    Ablator Community
                              </h3>

                           </div>

                           <div class="contents-grid">
                              <div class="contents-card community-card">
                                 <div class="card-title">
                                    <h5>
                                       Visit Ablator on Github
                                    </h5>
                                 </div>
                                 <div class="card-texts">
                                    <p>
                                       Ablator is an open-source project. Visit Ablator on Github to learn more and feel free
                                       to
                                       make your contributions.
                                    </p>
                                 </div>
                                 <div>
                                    <button class="custom-btn custom-btn-block"
                                       onclick="window.open('https://github.com/fostiropoulos/ablator')">Github
                                       Repository</button>
                                 </div>
                              </div>
                              <div class="contents-card community-card">
                                 <div class="card-title">
                                    <h5>
                                       Meet the developers
                                    </h5>
                                 </div>
                                 <div class="card-texts">
                                    <p>
                                       Ablator is developed and maintained by Deep USC Research Group from University of
                                       Southern California.
                                    </p>
                                 </div>
                                 <div>
                                    <button class="custom-btn custom-btn-block"
                                       onclick="window.open('https://deep.usc.edu')">DeepUSC Research Group</button>
                                 </div>
                              </div>
                           </div>
                        </div>
                  </div>
               </div>
            </body>
         </html>

.. only:: sidebar

   .. toctree::
      :maxdepth: 3
      :caption: Contents:

         Get started <notebooks/GettingStarted.ipynb>
         Basic Tutorials <tutorials>
         Intermediate Tutorials <intermediate_tutorials>
         API Reference <ablator.rst>
         More Example <notebooks/GettingStarted-more-demos.ipynb>

.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
