Example only. This folder is not deployed.

An Azure Functions timer trigger can run the same gmpp_pipeline.py script
once a quarter. The NCRONTAB expression 0 0 9 1 1,4,7,10 * means 09:00
UTC on the first day of January, April, July and October.

The function still cannot write straight to your Mac. You would copy or
sync the downloaded file into your GMPP folder afterwards, or use rclone
to Dropbox or OneDrive. The launchd job on the Mac is the simpler option
because it already has the files on the same computer as the training
script.
