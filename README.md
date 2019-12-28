# comprasnet_captcha_models

## Comprasnet and public procurement data in Brazil

In Brazil, Federal-level public procurements for ordinary goods and services must take place as online auctions. One of the most used platforms for such procurement auctions is [Comprasnet](https://comprasgovernamentais.gov.br/index.php/comprasnet-siasg).

There is an useful [API](http://compras.dados.gov.br/docs/home.html) for accessing data regarding procurement auctions, such as relevant dates, winning bids, and data regarding purchased goods and services, including detailed description and quantities.

Nevertheless, many other important information cannot be accessed via the API, and are available only in the procurement records (_atas_). Such information include the complete list of bids, with value, bidder and timestamps, detailed information about the auction phases and events, as well as all communication between bidders and the procurement official (_pregoeiro_).

The procurement records are available online as html pages on a different [address](http://comprasnet.gov.br/acesso.asp?url=/livre/pregao/ata0.asp), outside the API. If one wants to access information regarding a procurement auction, she must type in the search parameteres, navigate through several pages and, more importantly, solve a CAPTCHA. The process must be entirely repeated for every procurement auction.

This is not aligned with the principles of government transparency and makes it harder to analyze data from multiple procurement auctions.

## Types of captchas

After the user types in the search parameters and navigates through some pages, she must solve one five different types of captchas appear alternately in the webpage preceding a procurement report. These are:

bubble         | bubble_cut    | dotted         | dotted_wave  | wave          |
:-------------:|:-------------:|:-------------:|:-------------:|:-------------:
![alt text](https://github.com/andremenegatti/comprasnet_captcha_models/blob/master/captchas/test/bubble/1Q4CwZ_564.png "bubble captcha") | ![alt text](https://github.com/andremenegatti/comprasnet_captcha_models/blob/master/captchas/test/bubble_cut/14VXad_731.png "bubble_cut captcha") |![alt text](https://github.com/andremenegatti/comprasnet_captcha_models/blob/master/captchas/test/dotted/bTw31n_699.png "dotted captcha") | ![alt text](https://github.com/andremenegatti/comprasnet_captcha_models/blob/master/captchas/test/dotted_wave/Captcha000002.png "dotted_wave captcha") | ![alt text](https://github.com/andremenegatti/comprasnet_captcha_models/blob/master/captchas/test/wave/16CWaQ_701.png "wave captcha")

## Captcha-breaking CNNs

This repository contains python modules used for training and testing 4 convolutional neural networks (CNNs) designed to solve Comprasnet's captchas. It also includes training and testing data (_i.e._, captcha images) and the trained models (in .hdf5 files, with labels in separate .dat files).

Each CNN is built to handle a single captcha type. Each architecture was selected after a lot of testing and tweaking, but can definetely be further improved.

As of now, _dotted_wave_ captchas are not supported.

## Character segmentation

Since the models are trained to predict individual characters, character segmentation must be performed beforehand. Thus, the repository also includes python modules with simple algorithms for isolating the characters in captcha images.

The algorithms are specific to each captcha-type (with the exception of _bubble_ and _bubble_cut_, which are handled by a single algorithm).

Currently there is no algorithm for splitting _dotted_wave_ captchas, and this is the reason why there is no model for this captcha class. Help in this regard would be much appreciated.

## CNN for predicting captcha type

In addition to the models that perform character recognition, the repository also comprises a CNN to classify captchas according to the 5 types listed above.

Thus, one can use this model to predict a captcha's type and use this prediction to select which type-specific CNN to use.

## Automating data collection from Comprasnet's procurement records

This repository focuses on CNN training and testing. If you are interested in using these models for scraping Comprasnet's procurement records, check [comprasnet_captcha_breaker](https://github.com/andremenegatti/comprasnet_captcha_models).
